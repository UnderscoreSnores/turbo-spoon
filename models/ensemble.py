# =============================================================================
# models/ensemble.py - Ensemble prediction and signal generation
# =============================================================================

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    symbol: str
    date: datetime
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    predicted_return: float
    model_predictions: Dict


class EnsemblePredictor:
    """Ensemble model for generating trading signals"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.feature_engineer = None

    def load_models(self, model_dict: Dict):
        """Load trained models"""
        self.models = model_dict['models']
        self.scalers = model_dict['scalers']
        self.feature_columns = model_dict['feature_columns']

        # Import here to avoid circular imports
        from data.feature_engineering import FeatureEngineer
        self.feature_engineer = FeatureEngineer()

    def predict_single_symbol(self, df: pd.DataFrame, symbol: str,
                              date: pd.Timestamp) -> Dict[str, float]:
        """Generate predictions for one symbol"""
        if symbol not in self.models:
            return {}

        # Create features
        features_df = self.feature_engineer.create_all_features(df, symbol)

        if date not in features_df.index:
            return {}

        features = features_df.loc[date].fillna(0)
        predictions = {}

        for model_name, model in self.models[symbol].items():
            try:
                scaler = self.scalers[symbol][model_name]
                feature_cols = self.feature_columns[f"{model_name}_{symbol}"]

                X = features[feature_cols].values.reshape(1, -1)
                X_scaled = scaler.transform(X)

                pred = model.predict(X_scaled)[0]
                predictions[model_name] = pred

            except Exception as e:
                logger.error(f"Prediction failed for {symbol} {model_name}: {e}")

        return predictions

    def generate_signals(self, df: pd.DataFrame, date: pd.Timestamp,
                         symbols: List[str], min_confidence: float = 0.6) -> List[TradeSignal]:
        """Generate trading signals"""
        signals = []

        for symbol in symbols:
            predictions = self.predict_single_symbol(df, symbol, date)

            if not predictions:
                continue

            # Ensemble prediction
            pred_values = list(predictions.values())
            avg_prediction = np.mean(pred_values)
            pred_std = np.std(pred_values) if len(pred_values) > 1 else 0

            # Confidence based on prediction consistency
            confidence = 1 / (1 + pred_std) if pred_std > 0 else 1.0

            if confidence < min_confidence:
                continue

            # Generate signal
            if avg_prediction > 0.02:  # 2% threshold
                signal_type = 'BUY'
            elif avg_prediction < -0.02:
                signal_type = 'SELL'
            else:
                signal_type = 'HOLD'

            if signal_type != 'HOLD':
                signal = TradeSignal(
                    symbol=symbol,
                    date=date,
                    signal_type=signal_type,
                    confidence=confidence,
                    predicted_return=avg_prediction,
                    model_predictions=predictions
                )
                signals.append(signal)

        return signals
