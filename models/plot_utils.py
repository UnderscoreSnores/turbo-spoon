import plotly.graph_objs as go
import plotly.io as pio

def plot_interactive(prices, features, symbol, buy_signals=None, sell_signals=None, short_signals=None, cover_signals=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices.values, mode='lines', name='Price', line=dict(color='blue')))
    if "ma5" in features.columns:
        fig.add_trace(go.Scatter(x=features.index, y=features["ma5"], mode='lines', name='MA5', line=dict(color='orange', dash='dash')))
    if "ma20" in features.columns:
        fig.add_trace(go.Scatter(x=features.index, y=features["ma20"], mode='lines', name='MA20', line=dict(color='magenta', dash='dash')))
    if "macd" in features.columns:
        fig.add_trace(go.Scatter(x=features.index, y=features["macd"], mode='lines', name='MACD', line=dict(color='green', dash='dot')))
    if "rsi_14" in features.columns:
        fig.add_trace(go.Scatter(x=features.index, y=features["rsi_14"], mode='lines', name='RSI(14)', line=dict(color='red', dash='dot')))
    # Buy/Sell/Short/Cover signals
    if buy_signals:
        fig.add_trace(go.Scatter(x=[idx for idx, _ in buy_signals], y=[price for _, price in buy_signals], mode='markers', name='Buy', marker=dict(symbol='triangle-up', color='lime', size=12)))
    if sell_signals:
        fig.add_trace(go.Scatter(x=[idx for idx, _ in sell_signals], y=[price for _, price in sell_signals], mode='markers', name='Sell', marker=dict(symbol='triangle-down', color='red', size=12)))
    if short_signals:
        fig.add_trace(go.Scatter(x=[idx for idx, _ in short_signals], y=[price for _, price in short_signals], mode='markers', name='Short', marker=dict(symbol='x', color='black', size=12)))
    if cover_signals:
        fig.add_trace(go.Scatter(x=[idx for idx, _ in cover_signals], y=[price for _, price in cover_signals], mode='markers', name='Cover', marker=dict(symbol='circle', color='orange', size=12)))
    fig.update_layout(title=f"{symbol} - Interactive Trades & Indicators", xaxis_title="Date", yaxis_title="Price / Indicator Value", legend=dict(orientation="h"))
    return pio.to_html(fig, full_html=False)
