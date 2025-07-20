import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from pathlib import Path
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

def compute_rsi(prices, window=14):
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=window, min_periods=window).mean()
    ma_down = down.rolling(window=window, min_periods=window).mean()
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    return rsi

def add_technical_indicators(df, symbol):
    prices = df[symbol]
    if 'ma5' not in df.columns:
        df['ma5'] = prices.rolling(5).mean()
    if 'ma20' not in df.columns:
        df['ma20'] = prices.rolling(20).mean()
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    ma20 = prices.rolling(20).mean()
    std20 = prices.rolling(20).std()
    df['bb_upper'] = ma20 + 2 * std20
    df['bb_lower'] = ma20 - 2 * std20
    low14 = prices.rolling(14).min()
    high14 = prices.rolling(14).max()
    df['stoch_k'] = 100 * (prices - low14) / (high14 - low14)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    high = df['high'] if 'high' in df else prices
    low = df['low'] if 'low' in df else prices
    close = prices
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    if 'volume' in df:
        obv = (np.sign(prices.diff()) * df['volume']).fillna(0).cumsum()
        df['obv'] = obv
    for col in ['ma5', 'ma20', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'stoch_k', 'stoch_d', 'atr']:
        if col in df.columns:
            for lag in [1, 2, 3]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
    df['range'] = high - low
    df['gap'] = prices - prices.shift(1)
    return df

def make_multiclass_target(prices, up_thresh=0.01, down_thresh=-0.01):
    returns = prices.pct_change().shift(-1)
    target = pd.Series(0, index=returns.index)
    target[returns > up_thresh * 2] = 2
    target[(returns > up_thresh) & (returns <= up_thresh * 2)] = 1
    target[(returns < down_thresh) & (returns >= down_thresh * 2)] = -1
    target[returns < down_thresh * 2] = -2
    return target

def get_buy_strength(test_acc, model_return_pct):
    if test_acc > 0.7 and model_return_pct > 20:
        return 5, "Strong Buy"
    elif test_acc > 0.6 and model_return_pct > 10:
        return 4, "Decent Buy"
    elif test_acc > 0.5 and model_return_pct > 0:
        return 3, "Buy"
    elif test_acc > 0.4:
        return 2, "Weak (Short)"
    else:
        return 1, "Short"

class ModelTrainer:
    def __init__(self, output_dir: str = "models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.model_scores = {}

    def get_model_configs(self):
        return {
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42,
                    'n_jobs': -1,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'class_weight': 'balanced'
                }
            },
            'xgboost': {
                'model': XGBClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbosity': 0,
                    'use_label_encoder': False,
                    'eval_metric': 'mlogloss'
                }
            },
            'lightgbm': {
                'model': LGBMClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbosity': -1,
                    'class_weight': 'balanced'
                }
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'random_state': 42,
                    'max_iter': 1000,
                    'class_weight': 'balanced'
                }
            },
            'mlp': {
                'model': MLPClassifier,
                'params': {
                    'random_state': 42,
                    'max_iter': 500
                }
            }
        }

    def prepare_features(self, df, symbol):
        if 'ma5' not in df.columns:
            df['ma5'] = df[symbol].rolling(5).mean()
        if 'ma20' not in df.columns:
            df['ma20'] = df[symbol].rolling(20).mean()
        df = add_technical_indicators(df, symbol)
        return df

    def prepare_training_data(self, features_df: pd.DataFrame, symbol: str, train_end_date: str, multiclass=True):
        features_df = self.prepare_features(features_df, symbol)
        if multiclass:
            target = make_multiclass_target(features_df[symbol])
        else:
            target = (features_df[symbol].shift(-1) > features_df[symbol]).astype(int)
        features_df["target"] = target
        features_df = features_df[features_df["target"].notna()]
        features_df["target"] = features_df["target"].astype(int)
        X = features_df.drop(columns=["target"])
        y = features_df["target"]
        train_end = pd.to_datetime(train_end_date)
        train_mask = X.index <= train_end
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_train = X_train.fillna(0)
        X_train = X_train.replace([np.inf, -np.inf], 0)
        constant_cols = X_train.columns[X_train.std() == 0]
        if len(constant_cols) > 0:
            X_train = X_train.drop(columns=constant_cols)
            logger.info(f"Removed {len(constant_cols)} constant columns for {symbol}")
        return X_train, y_train

    def train_single_symbol(self, features_df: pd.DataFrame, symbol: str, train_end_date: str, multiclass=True):
        try:
            X_train, y_train = self.prepare_training_data(features_df, symbol, train_end_date, multiclass=multiclass)
            if len(X_train) < 100:
                logger.warning(f"Insufficient training data for {symbol}: {len(X_train)} samples")
                return
            logger.info(f"Training {symbol} with {len(X_train)} samples, {X_train.shape[1]} features")
            self.models[symbol] = {}
            self.scalers[symbol] = {}
            self.model_scores[symbol] = {}
            configs = self.get_model_configs()
            voting_estimators = []
            for model_name, config in configs.items():
                try:
                    logger.info(f"Training {model_name} for {symbol}")
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    model = config['model'](**config['params'])
                    model.fit(X_train_scaled, y_train)
                    self.models[symbol][model_name] = model
                    self.scalers[symbol][model_name] = scaler
                    self.feature_columns[f"{symbol}_{model_name}"] = X_train.columns.tolist()
                    voting_estimators.append((model_name, model))
                    tscv = TimeSeriesSplit(n_splits=3)
                    cv_scores = cross_val_score(
                        model, X_train_scaled, y_train,
                        cv=tscv,
                        scoring='accuracy',
                        n_jobs=1
                    )
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    y_pred = model.predict(X_train_scaled)
                    in_sample_acc = accuracy_score(y_train, y_pred)
                    self.model_scores[symbol][model_name] = {
                        'cv_acc_mean': cv_mean,
                        'cv_acc_std': cv_std,
                        'in_sample_acc': in_sample_acc,
                        'n_features': X_train.shape[1],
                        'n_samples': len(X_train)
                    }
                    logger.info(f"{symbol} {model_name} - CV Acc: {cv_mean:.3f}±{cv_std:.3f}, In-sample Acc: {in_sample_acc:.3f}")
                except Exception as model_error:
                    logger.error(f"Training failed for {symbol} {model_name}: {model_error}")
                    continue
            if voting_estimators:
                voting_clf = VotingClassifier(estimators=voting_estimators, voting='soft')
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                voting_clf.fit(X_train_scaled, y_train)
                self.models[symbol]['voting'] = voting_clf
                self.scalers[symbol]['voting'] = scaler
                self.feature_columns[f"{symbol}_voting"] = X_train.columns.tolist()
        except Exception as e:
            logger.error(f"Failed to train models for {symbol}: {e}")

    def save_model(self, symbol):
        symbol_dir = self.output_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        for model_name in self.models[symbol]:
            model_path = symbol_dir / f"{model_name}_model.joblib"
            joblib.dump(self.models[symbol][model_name], model_path)
            scaler_path = symbol_dir / f"{model_name}_scaler.joblib"
            joblib.dump(self.scalers[symbol][model_name], scaler_path)
            feature_path = symbol_dir / f"{model_name}_features.joblib"
            joblib.dump(self.feature_columns[f"{symbol}_{model_name}"], feature_path)

    def load_model(self, symbol):
        symbol_dir = self.output_dir / symbol
        if not symbol_dir.exists():
            print(f"No saved models found for {symbol}")
            return
        self.models[symbol] = {}
        self.scalers[symbol] = {}
        for model_file in symbol_dir.glob("*_model.joblib"):
            model_name = model_file.stem.replace("_model", "")
            self.models[symbol][model_name] = joblib.load(model_file)
            scaler_file = symbol_dir / f"{model_name}_scaler.joblib"
            if scaler_file.exists():
                self.scalers[symbol][model_name] = joblib.load(scaler_file)
            feature_file = symbol_dir / f"{model_name}_features.joblib"
            if feature_file.exists():
                self.feature_columns[f"{symbol}_{model_name}"] = joblib.load(feature_file)

    def get_trained_models(self):
        return {
            'models': self.models,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'scores': self.model_scores
        }

    def get_model_summary(self):
        summary = []
        for symbol in self.models:
            for model_name in self.models[symbol]:
                scores = self.model_scores.get(symbol, {}).get(model_name, {})
                summary.append({
                    'symbol': symbol,
                    'model': model_name,
                    'cv_acc_mean': scores.get('cv_acc_mean', scores.get('in_sample_acc', 'N/A')),
                    'n_features': scores.get('n_features', 'N/A'),
                    'n_samples': scores.get('n_samples', 'N/A')
                })
        return pd.DataFrame(summary)
