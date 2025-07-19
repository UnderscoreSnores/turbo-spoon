import pandas as pd
import yfinance as yf
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class NASDAQHistoricalPipeline:
    def __init__(self):
        # NASDAQ 100 symbols (you can customize this list)
        self.symbols = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'NVDA',
            'PYPL', 'ADBE', 'NFLX', 'INTC', 'CMCSA', 'PEP', 'CSCO', 'AVGO',
            'TXN', 'QCOM', 'COST', 'AMGN', 'SBUX', 'INTU', 'BKNG', 'ISRG',
            'GILD', 'MDLZ', 'ADP', 'VRTX', 'FISV', 'REGN', 'ATVI', 'CSX',
            'ILMN', 'MU', 'AMAT', 'ADI', 'MELI', 'LRCX', 'EXC', 'JD'
        ]

    def run_pipeline(self, start_date: str) -> Tuple[pd.DataFrame, List[str]]:
        """Fetch historical data for all symbols"""
        logger.info(f"Fetching data from {start_date}")

        data = {}
        successful_symbols = []

        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date)
                if len(hist) > 100:  # Ensure sufficient data
                    data[symbol] = hist['Close']
                    successful_symbols.append(symbol)
                    logger.info(f"Successfully fetched {symbol}")
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")

        df = pd.DataFrame(data)
        logger.info(f"Fetched data for {len(successful_symbols)} symbols")

        return df, successful_symbols