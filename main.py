# =============================================================================
# main.py - Main orchestration and workflow
# =============================================================================

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import json

from data.fetch_data import NASDAQHistoricalPipeline
from data.feature_engineering import FeatureEngineer
from models.train import ModelTrainer
from models.ensemble import EnsemblePredictor
from models.evaluate import BacktestEngine
from utils.plot import TradingVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AITrader:
    """Main AI Trading Application"""

    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.data_pipeline = NASDAQHistoricalPipeline()
        self.feature_engineer = FeatureEngineer()
        self.trainer = ModelTrainer()
        self.ensemble = EnsemblePredictor()
        self.backtester = BacktestEngine()
        self.visualizer = TradingVisualizer()

        self.df = None

    def _load_config(self, config_path: str) -> dict:
        """Load configuration"""
        default_config = {
            "data": {
                "start_date": "2023-01-01",
                "symbols_limit": 50
            },
            "training": {
                "train_end_date": "2023-12-31",
                "models": ["random_forest", "xgboost", "lightgbm", "ridge"]
            },
            "backtesting": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 100000,
                "max_positions": 20
            }
        }

        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        except FileNotFoundError:
            logger.info("No config file found, using defaults")

        return default_config

    def run_full_pipeline(self):
        """Execute complete trading pipeline"""
        logger.info("Starting AI Trader Pipeline")

        # 1. Data Collection
        logger.info("Step 1: Data Collection")
        self.df, _ = self.data_pipeline.run_pipeline(
            start_date=self.config["data"]["start_date"]
        )

        # 2. Symbol Selection
        logger.info("Step 2: Symbol Selection")
        symbol_coverage = self.df.notna().sum().sort_values(ascending=False)
        symbols = symbol_coverage.head(self.config["data"]["symbols_limit"]).index.tolist()
        logger.info(f"Selected {len(symbols)} symbols")

        # 3. Feature Engineering
        logger.info("Step 3: Feature Engineering")
        features_dict = {}
        for symbol in symbols:
            features_dict[symbol] = self.feature_engineer.create_all_features(self.df, symbol)

        # 4. Model Training
        logger.info("Step 4: Model Training")
        self.trainer.train_multiple_symbols(
            features_dict=features_dict,
            symbols=symbols,
            train_end_date=self.config["training"]["train_end_date"]
        )

        # 5. Ensemble Setup
        logger.info("Step 5: Ensemble Setup")
        self.ensemble.load_models(self.trainer.get_trained_models())

        # 6. Backtesting
        logger.info("Step 6: Backtesting")
        results = self.backtester.run_backtest(
            df=self.df,
            ensemble=self.ensemble,
            symbols=symbols[:20],  # Test on subset
            **self.config["backtesting"]
        )

        # 7. Visualization
        logger.info("Step 7: Results Visualization")
        self.visualizer.plot_backtest_results(results)

        # 8. Live Signals
        logger.info("Step 8: Generate Live Signals")
        latest_signals = self.ensemble.generate_signals(
            df=self.df,
            date=self.df.index[-1],
            symbols=symbols[:10]
        )

        self._print_results(results, latest_signals)

    def _print_results(self, results, signals):
        """Print summary results"""
        print("\n" + "=" * 50)
        print("AI TRADER RESULTS")
        print("=" * 50)
        print(f"Total Return: {results.total_return:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")
        print(f"Win Rate: {results.win_rate:.2%}")

        print(f"\nLive Signals ({len(signals)}):")
        for signal in signals:
            print(f"  {signal.symbol}: {signal.signal_type} "
                  f"(Conf: {signal.confidence:.2f})")


if __name__ == "__main__":
    trader = AITrader()
    trader.run_full_pipeline()
