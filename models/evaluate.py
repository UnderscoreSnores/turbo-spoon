# =============================================================================
# models/evaluate.py - Backtesting and evaluation
# =============================================================================

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    portfolio_history: pd.DataFrame
    trades: pd.DataFrame


class BacktestEngine:
    """Backtesting engine for trading strategies"""

    def run_backtest(self, df: pd.DataFrame, ensemble, symbols: List[str],
                     start_date: str, end_date: str, initial_capital: float = 100000,
                     max_positions: int = 20, transaction_cost: float = 0.001) -> BacktestResults:
        """Run comprehensive backtest"""
        logger.info(f"Running backtest: {start_date} to {end_date}")

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Initialize portfolio
        portfolio = {'cash': initial_capital, 'positions': {}, 'history': []}
        trades = []

        trading_dates = [d for d in df.index if start_dt <= d <= end_dt]

        for date in trading_dates:
            # Generate signals
            signals = ensemble.generate_signals(df, date, symbols)

            # Calculate portfolio value
            portfolio_value = portfolio['cash']
            for sym, shares in portfolio['positions'].items():
                if sym in df.columns and date in df.index:
                    price = df.loc[date, sym]
                    if not pd.isna(price):
                        portfolio_value += shares * price

            # Execute trades
            for signal in signals[:max_positions]:
                if signal.symbol not in df.columns or date not in df.index:
                    continue

                current_price = df.loc[date, signal.symbol]
                if pd.isna(current_price) or current_price <= 0:
                    continue

                position_size = portfolio_value * 0.05  # 5% position size

                if signal.signal_type == 'BUY':
                    shares_to_buy = int(position_size / current_price)
                    cost = shares_to_buy * current_price * (1 + transaction_cost)

                    if cost <= portfolio['cash'] and shares_to_buy > 0:
                        portfolio['cash'] -= cost
                        portfolio['positions'][signal.symbol] = \
                            portfolio['positions'].get(signal.symbol, 0) + shares_to_buy

                        trades.append({
                            'date': date, 'symbol': signal.symbol, 'action': 'BUY',
                            'shares': shares_to_buy, 'price': current_price,
                            'confidence': signal.confidence
                        })

                elif (signal.signal_type == 'SELL' and
                      signal.symbol in portfolio['positions']):
                    shares_to_sell = portfolio['positions'][signal.symbol]
                    if shares_to_sell > 0:
                        proceeds = shares_to_sell * current_price * (1 - transaction_cost)
                        portfolio['cash'] += proceeds
                        del portfolio['positions'][signal.symbol]

                        trades.append({
                            'date': date, 'symbol': signal.symbol, 'action': 'SELL',
                            'shares': shares_to_sell, 'price': current_price,
                            'confidence': signal.confidence
                        })

            # Record history
            portfolio['history'].append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': portfolio['cash'],
                'positions': len(portfolio['positions'])
            })

        # Calculate performance metrics
        history_df = pd.DataFrame(portfolio['history'])
        trades_df = pd.DataFrame(trades)

        returns = history_df['portfolio_value'].pct_change().dropna()
        total_return = (history_df['portfolio_value'].iloc[-1] / initial_capital) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        # Max drawdown
        running_max = history_df['portfolio_value'].expanding().max()
        drawdown = (history_df['portfolio_value'] - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Trade statistics
        win_rate = 0
        avg_trade_return = 0
        if len(trades_df) > 0:
            # Simplified trade return calculation
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']

            trade_returns = []
            for _, sell in sell_trades.iterrows():
                matching_buys = buy_trades[buy_trades['symbol'] == sell['symbol']]
                if len(matching_buys) > 0:
                    buy_price = matching_buys.iloc[-1]['price']
                    trade_return = (sell['price'] - buy_price) / buy_price
                    trade_returns.append(trade_return)

            if trade_returns:
                win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
                avg_trade_return = np.mean(trade_returns)

        return BacktestResults(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            avg_trade_return=avg_trade_return,
            portfolio_history=history_df,
            trades=trades_df
        )
