# data/fetch_data.py

import pandas as pd

def fetch_price_data_from_csv(df, symbol, start, end):
    """
    df: DataFrame of closes (dates as index, symbols as columns)
    symbol: ticker symbol (string)
    start, end: date strings or pd.Timestamp
    Returns: Series of closes for the symbol in the date range
    """
    if symbol not in df.columns:
        return pd.Series(dtype=float)
    return df.loc[start:end, symbol].dropna()
