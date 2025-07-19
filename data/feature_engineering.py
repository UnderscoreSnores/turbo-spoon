# =============================================================================
# data/feature_engineering.py - Enhanced version for your modular structure
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Technical indicators implementation"""

    @staticmethod
    def sma(prices, period):
        return prices.rolling(window=period).mean()

    @staticmethod
    def ema(prices, period):
        return prices.ewm(span=period).mean()

    @staticmethod
    def rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def bollinger_bands(prices, period=20, std_dev=2):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    @staticmethod
    def macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram


class FeatureEngineer:
    """Advanced feature engineering for trading"""

    def __init__(self):
        self.indicators = TechnicalIndicators()

    def create_all_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create comprehensive feature set for a symbol"""
        logger.info(f"Creating features for {symbol}")

        # Price-based features
        price_features = self._create_price_features(df, symbol)

        # Technical indicators
        tech_features = self._create_technical_features(df, symbol)

        # Market-wide features
        market_features = self._create_market_features(df)

        # Combine all features
        all_features = pd.concat([price_features, tech_features, market_features], axis=1)

        # Add lagged features
        for lag in [1, 2, 3, 5]:
            lagged = price_features.shift(lag).add_suffix(f'_lag{lag}')
            all_features = pd.concat([all_features, lagged], axis=1)

        return all_features.dropna()

    def _create_price_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Price-based features"""
        features = df[[symbol]].copy()
        price = features[symbol]

        # Returns at different horizons
        for period in [1, 2, 3, 5, 10, 20]:
            features[f'{symbol}_return_{period}d'] = price.pct_change(period)
            features[f'{symbol}_log_return_{period}d'] = np.log(price / price.shift(period))

        # Price ratios
        for period in [5, 10, 20, 50, 200]:
            features[f'{symbol}_price_ratio_{period}d'] = price / price.rolling(period).mean()

        # Volatility features
        for period in [5, 10, 20]:
            returns = price.pct_change()
            features[f'{symbol}_volatility_{period}d'] = returns.rolling(period).std()

        return features

    def _create_technical_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Technical indicator features"""
        features = pd.DataFrame(index=df.index)
        price = df[symbol]

        # Moving averages and ratios
        for period in [5, 10, 20, 50]:
            sma = self.indicators.sma(price, period)
            features[f'{symbol}_sma_{period}'] = sma
            features[f'{symbol}_price_sma_ratio_{period}'] = price / sma

        # RSI
        features[f'{symbol}_rsi_14'] = self.indicators.rsi(price, 14)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(price)
        features[f'{symbol}_bb_position'] = (price - bb_lower) / (bb_upper - bb_lower)

        # MACD
        macd, signal, histogram = self.indicators.macd(price)
        features[f'{symbol}_macd'] = macd
        features[f'{symbol}_macd_histogram'] = histogram

        return features

    def _create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market-wide features"""
        features = pd.DataFrame(index=df.index)

        # Market proxy (average of all stocks)
        market_prices = df.mean(axis=1, skipna=True)
        features['market_return_1d'] = market_prices.pct_change()
        features['market_return_5d'] = market_prices.pct_change(5)

        # Cross-sectional features
        daily_returns = df.pct_change()
        features['stocks_positive_pct'] = (daily_returns > 0).mean(axis=1, skipna=True)
        features['return_dispersion'] = daily_returns.std(axis=1, skipna=True)

        return features
