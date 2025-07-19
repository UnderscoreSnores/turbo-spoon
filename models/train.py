# =============================================================================
# models/train.py - Model training module
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
from typing import Dict, List
import logging
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Model training and management"""

    def __init__(self, output_dir: str = "models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.model_scores = {}

    def get_model_configs(self):
        """Default model configurations"""
        return {
            'random_forest': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42,
                    'n_jobs': -1,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2
                }
            },
            'xgboost': {
                'model': xgb.XGBRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbosity': 0
                }
            },
            'lightgbm': {
                'model': lgb.LGBMRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbosity': -1
                }
            },
            'ridge': {
                'model': Ridge,
                'params': {
                    'alpha': 1.0,
                    'random_state': 42
                }
            }
        }

    def create_target_variable(self, features_df: pd.DataFrame, symbol: str,
                               target_horizon: int = 1) -> pd.Series:
        """Create target variable (future returns)"""
        # Look for return column in features
        return_col = f'{symbol}_return_{target_horizon}d'

        if return_col in features_df.columns:
            # Shift returns to get future returns
            target = features_df[return_col].shift(-target_horizon)
        else:
            # Fallback: calculate returns from price if available
            if symbol in features_df.columns:
                price = features_df[symbol]
                target = price.pct_change(target_horizon).shift(-target_horizon)
            else:
                raise ValueError(f"Cannot create target for {symbol}")

        return target

    def prepare_training_data(self, features_df: pd.DataFrame, symbol: str,
                              train_end_date: str, target_horizon: int = 1):
        """Prepare training data with proper alignment"""
        # Create target
        target = self.create_target_variable(features_df, symbol, target_horizon)

        # Align features and target
        common_index = features_df.index.intersection(target.index)
        X = features_df.loc[common_index]
        y = target.loc[common_index]

        # Remove rows where target is NaN
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]

        # Filter for training period
        train_end = pd.to_datetime(train_end_date)
        train_mask = X.index <= train_end
        X_train = X[train_mask]
        y_train = y[train_mask]

        # Clean features (fill NaN, remove inf)
        X_train = X_train.fillna(0)
        X_train = X_train.replace([np.inf, -np.inf], 0)

        # Remove constant columns
        constant_cols = X_train.columns[X_train.std() == 0]
        if len(constant_cols) > 0:
            X_train = X_train.drop(columns=constant_cols)
            logger.info(f"Removed {len(constant_cols)} constant columns for {symbol}")

        return X_train, y_train

    def train_single_symbol(self, features_df: pd.DataFrame, symbol: str,
                            train_end_date: str, target_horizon: int = 1):
        """Train models for a single symbol"""
        try:
            # Prepare data
            X_train, y_train = self.prepare_training_data(
                features_df, symbol, train_end_date, target_horizon
            )

            if len(X_train) < 100:
                logger.warning(f"Insufficient training data for {symbol}: {len(X_train)} samples")
                return

            logger.info(f"Training {symbol} with {len(X_train)} samples, {X_train.shape[1]} features")

            # Initialize storage for this symbol
            self.models[symbol] = {}
            self.scalers[symbol] = {}
            self.model_scores[symbol] = {}

            # Train each model type
            configs = self.get_model_configs()

            for model_name, config in configs.items():
                try:
                    logger.info(f"Training {model_name} for {symbol}")

                    # Scale features
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)

                    # Initialize and train model
                    model = config['model'](**config['params'])
                    model.fit(X_train_scaled, y_train)

                    # Store model and scaler
                    self.models[symbol][model_name] = model
                    self.scalers[symbol][model_name] = scaler
                    self.feature_columns[f"{symbol}_{model_name}"] = X_train.columns.tolist()

                    # Cross-validation
                    try:
                        cv_scores = cross_val_score(
                            model, X_train_scaled, y_train,
                            cv=TimeSeriesSplit(n_splits=3),
                            scoring='r2',
                            n_jobs=1  # Avoid nested parallelism
                        )
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()

                        # In-sample score
                        y_pred = model.predict(X_train_scaled)
                        in_sample_r2 = r2_score(y_train, y_pred)
                        in_sample_mse = mean_squared_error(y_train, y_pred)

                        # Store scores
                        self.model_scores[symbol][model_name] = {
                            'cv_r2_mean': cv_mean,
                            'cv_r2_std': cv_std,
                            'in_sample_r2': in_sample_r2,
                            'in_sample_mse': in_sample_mse,
                            'n_features': X_train.shape[1],
                            'n_samples': len(X_train)
                        }

                        logger.info(f"{symbol} {model_name} - CV R²: {cv_mean:.3f}±{cv_std:.3f}, "
                                    f"In-sample R²: {in_sample_r2:.3f}")

                    except Exception as cv_error:
                        logger.warning(f"CV failed for {symbol} {model_name}: {cv_error}")
                        # Store basic scores without CV
                        y_pred = model.predict(X_train_scaled)
                        self.model_scores[symbol][model_name] = {
                            'in_sample_r2': r2_score(y_train, y_pred),
                            'in_sample_mse': mean_squared_error(y_train, y_pred),
                            'n_features': X_train.shape[1],
                            'n_samples': len(X_train)
                        }

                except Exception as model_error:
                    logger.error(f"Training failed for {symbol} {model_name}: {model_error}")
                    continue

            logger.info(f"Successfully trained {len(self.models[symbol])} models for {symbol}")

        except Exception as e:
            logger.error(f"Failed to train models for {symbol}: {e}")

    def train_multiple_symbols(self, features_dict: Dict, symbols: List[str],
                               train_end_date: str, target_horizon: int = 1):
        """Train models for multiple symbols"""
        logger.info(f"Training models for {len(symbols)} symbols")

        successful_symbols = []
        failed_symbols = []

        for i, symbol in enumerate(symbols, 1):
            logger.info(f"Processing symbol {i}/{len(symbols)}: {symbol}")

            if symbol in features_dict:
                try:
                    self.train_single_symbol(
                        features_dict[symbol], symbol, train_end_date, target_horizon
                    )
                    if symbol in self.models and self.models[symbol]:
                        successful_symbols.append(symbol)
                    else:
                        failed_symbols.append(symbol)
                except Exception as e:
                    logger.error(f"Failed to process {symbol}: {e}")
                    failed_symbols.append(symbol)
            else:
                logger.warning(f"No features found for {symbol}")
                failed_symbols.append(symbol)

        logger.info(f"Training completed: {len(successful_symbols)} successful, "
                    f"{len(failed_symbols)} failed")

        if successful_symbols:
            self.save_models()
            self.save_training_report()

        return successful_symbols, failed_symbols

    def save_models(self):
        """Save all trained models and scalers"""
        logger.info("Saving trained models...")

        for symbol in self.models:
            symbol_dir = self.output_dir / symbol
            symbol_dir.mkdir(exist_ok=True)

            for model_name, model in self.models[symbol].items():
                # Save model
                model_path = symbol_dir / f'{model_name}_model.joblib'
                joblib.dump(model, model_path)

                # Save scaler
                scaler_path = symbol_dir / f'{model_name}_scaler.joblib'
                joblib.dump(self.scalers[symbol][model_name], scaler_path)

                # Save feature columns
                feature_path = symbol_dir / f'{model_name}_features.joblib'
                joblib.dump(self.feature_columns[f"{symbol}_{model_name}"], feature_path)

        logger.info(f"Models saved to {self.output_dir}")

    def save_training_report(self):
        """Save training performance report"""
        report_data = []

        for symbol in self.model_scores:
            for model_name, scores in self.model_scores[symbol].items():
                report_data.append({
                    'symbol': symbol,
                    'model': model_name,
                    **scores
                })

        if report_data:
            report_df = pd.DataFrame(report_data)
            report_path = self.output_dir / 'training_report.csv'
            report_df.to_csv(report_path, index=False)
            logger.info(f"Training report saved to {report_path}")

    def load_models(self, symbol: str = None):
        """Load previously saved models"""
        if symbol:
            symbols_to_load = [symbol]
        else:
            symbols_to_load = [d.name for d in self.output_dir.iterdir() if d.is_dir()]

        for sym in symbols_to_load:
            symbol_dir = self.output_dir / sym
            if not symbol_dir.exists():
                continue

            self.models[sym] = {}
            self.scalers[sym] = {}

            model_files = list(symbol_dir.glob('*_model.joblib'))
            for model_file in model_files:
                model_name = model_file.stem.replace('_model', '')

                # Load model
                self.models[sym][model_name] = joblib.load(model_file)

                # Load scaler
                scaler_file = symbol_dir / f'{model_name}_scaler.joblib'
                if scaler_file.exists():
                    self.scalers[sym][model_name] = joblib.load(scaler_file)

                # Load features
                feature_file = symbol_dir / f'{model_name}_features.joblib'
                if feature_file.exists():
                    self.feature_columns[f"{sym}_{model_name}"] = joblib.load(feature_file)

        logger.info(f"Loaded models for {len(self.models)} symbols")

    def get_trained_models(self):
        """Return trained models dictionary"""
        return {
            'models': self.models,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'scores': self.model_scores
        }

    def get_model_summary(self):
        """Get summary of trained models"""
        summary = []
        for symbol in self.models:
            for model_name in self.models[symbol]:
                scores = self.model_scores.get(symbol, {}).get(model_name, {})
                summary.append({
                    'symbol': symbol,
                    'model': model_name,
                    'r2_score': scores.get('cv_r2_mean', scores.get('in_sample_r2', 'N/A')),
                    'n_features': scores.get('n_features', 'N/A'),
                    'n_samples': scores.get('n_samples', 'N/A')
                })

        return pd.DataFrame(summary)