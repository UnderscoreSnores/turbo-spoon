# =============================================================================
# utils/plot.py - Visualization utilities
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


class TradingVisualizer:
    """Visualization tools for trading results"""

    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        plt.style.use('seaborn-v0_8')

    def plot_backtest_results(self, results):
        """Plot comprehensive backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Portfolio value over time
        axes[0, 0].plot(results.portfolio_history['date'],
                        results.portfolio_history['portfolio_value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')

        # Drawdown
        portfolio_values = results.portfolio_history['portfolio_value']
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max

        axes[0, 1].fill_between(results.portfolio_history['date'], drawdown, 0, alpha=0.7)
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown (%)')

        # Trade distribution
        if len(results.trades) > 0:
            axes[1, 0].hist(results.trades.groupby('symbol').size(), bins=20)
            axes[1, 0].set_title('Trades per Symbol')
            axes[1, 0].set_xlabel('Number of Trades')

        # Performance metrics text
        metrics_text = f"""
        Total Return: {results.total_return:.2%}
        Annual Return: {results.annual_return:.2%}
        Volatility: {results.volatility:.2%}
        Sharpe Ratio: {results.sharpe_ratio:.2f}
        Max Drawdown: {results.max_drawdown:.2%}
        Win Rate: {results.win_rate:.2%}
        Total Trades: {results.total_trades}
        """

        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()