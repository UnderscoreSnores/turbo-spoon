import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import threading
import sys
import os
import shutil
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from data.fetch_data import fetch_price_data_from_csv
from models.train import ModelTrainer, compute_rsi, add_technical_indicators
from sklearn.metrics import accuracy_score

PLOTS_DIR = "plots"
if os.path.exists(PLOTS_DIR):
    shutil.rmtree(PLOTS_DIR)
os.makedirs(PLOTS_DIR, exist_ok=True)
stop_flag = False

def listen_for_q():
    global stop_flag
    try:
        while True:
            if sys.stdin.read(1).lower() == 'q':
                stop_flag = True
                print("\nDetected 'q'. Stopping after current symbol...\n")
                break
    except Exception:
        pass

def plot_trades_and_indicators(prices, X_test_idx, y_pred, symbol, best_model_name, features):
    buy_signals = []
    sell_signals = []
    short_signals = []
    cover_signals = []
    position_type = None
    for i, idx in enumerate(X_test_idx):
        pred = y_pred[i]
        price = prices.loc[idx]
        if pred == 1 and position_type is None:
            buy_signals.append((idx, price))
            position_type = "long"
        elif pred == 0 and position_type is None:
            short_signals.append((idx, price))
            position_type = "short"
        elif position_type == "long" and pred == 0:
            sell_signals.append((idx, price))
            position_type = None
        elif position_type == "short" and pred == 1:
            cover_signals.append((idx, price))
            position_type = None
    if position_type == "long":
        sell_signals.append((X_test_idx[-1], prices.loc[X_test_idx[-1]]))
    elif position_type == "short":
        cover_signals.append((X_test_idx[-1], prices.loc[X_test_idx[-1]]))

    plt.figure(figsize=(16, 8))
    plt.plot(prices.loc[X_test_idx], label='Price', color='blue', linewidth=1.5)
    if "ma5" in features.columns:
        plt.plot(features.loc[X_test_idx, "ma5"], label='MA5', color='orange', linestyle='--')
    if "ma20" in features.columns:
        plt.plot(features.loc[X_test_idx, "ma20"], label='MA20', color='magenta', linestyle='--')
    if "macd" in features.columns:
        plt.plot(features.loc[X_test_idx, "macd"], label='MACD', color='green', linestyle=':')
    if "rsi_14" in features.columns:
        plt.plot(features.loc[X_test_idx, "rsi_14"], label='RSI(14)', color='red', linestyle=':')
    if buy_signals:
        buy_idx, buy_price = zip(*buy_signals)
        plt.scatter(buy_idx, buy_price, marker='^', color='lime', label='Buy', s=120, zorder=5)
    if sell_signals:
        sell_idx, sell_price = zip(*sell_signals)
        plt.scatter(sell_idx, sell_price, marker='v', color='red', label='Sell', s=120, zorder=5)
    if short_signals:
        short_idx, short_price = zip(*short_signals)
        plt.scatter(short_idx, short_price, marker='x', color='black', label='Short', s=120, zorder=5)
    if cover_signals:
        cover_idx, cover_price = zip(*cover_signals)
        plt.scatter(cover_idx, cover_price, marker='o', color='orange', label='Cover', s=120, zorder=5)
    plt.title(f"{symbol} - Trades & Indicators by {best_model_name}")
    plt.xlabel("Date")
    plt.ylabel("Price / Indicator Value")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{symbol.lower()}_trades_indicators.png"))
    plt.close()

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

def process_symbol(symbol, df, trainer):
    global stop_flag
    if stop_flag:
        return None
    print(f"Processing {symbol} ...", flush=True)
    prices = fetch_price_data_from_csv(df, symbol, "2023-01-01", "2025-07-19")
    if stop_flag:
        return None
    if len(prices) < 100 or prices.iloc[-1] < 1:
        print(f"Skipping {symbol}: not enough price data or price < 1")
        return None
    features = pd.DataFrame({symbol: prices})
    features['ma5'] = prices.rolling(5).mean()
    features['ma20'] = prices.rolling(20).mean()
    features = add_technical_indicators(features, symbol)
    features["target"] = (prices.shift(-1) > prices).astype(int)
    features = features[features["target"].notna()]
    features["target"] = features["target"].astype(int)
    print(f"{symbol}: features shape after dropna: {features.shape}")
    split_date = "2024-01-01"
    feature_cols = [col for col in features.columns if col != "target"]
    X_train = features.loc[:split_date, feature_cols]
    y_train = features.loc[:split_date, "target"]
    X_test = features.loc[split_date:, feature_cols]
    y_test = features.loc[split_date:, "target"]
    if stop_flag:
        return None
    if len(X_train) < 30 or len(X_test) < 5:
        print(f"Skipping {symbol}: not enough train/test data (train: {len(X_train)}, test: {len(X_test)})")
        return None

    trainer.train_single_symbol(features, symbol, train_end_date=split_date, multiclass=False)

    best_model_name = None
    best_test_acc = -np.inf
    best_model = None
    best_scaler = None
    best_y_pred = None
    best_feature_cols = None

    for model_name in trainer.models.get(symbol, {}):
        model = trainer.models[symbol][model_name]
        scaler = trainer.scalers[symbol][model_name]
        model_feature_cols = trainer.feature_columns[f"{symbol}_{model_name}"]
        X_test_model = X_test[model_feature_cols].fillna(0)
        y_pred = model.predict(scaler.transform(X_test_model))
        test_acc = accuracy_score(y_test, y_pred)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_name = model_name
            best_model = model
            best_scaler = scaler
            best_y_pred = y_pred
            best_feature_cols = model_feature_cols

    if best_model is None:
        print(f"Skipping {symbol}: no model could be trained")
        return None

    # Simulate model trading using best model, with 10% stop-loss for both long and short, skip weekends
    y_pred = best_y_pred
    initial_cash = 1000
    cash = initial_cash
    position = 0
    position_type = None
    trade_count = 0
    entry_price = None
    stop_loss_pct = 0.10
    predicted_sell_point = None
    predicted_sell_price = None

    for i, idx in enumerate(X_test.index):
        if idx.weekday() >= 5:
            continue
        pred = y_pred[i]
        price = prices.loc[idx]
        if pred == 1 and position_type is None:
            position = cash / price
            cash = 0
            position_type = "long"
            entry_price = price
            trade_count += 1
            if predicted_sell_point is None:
                buy_idx = idx
        elif pred == 0 and position_type is None:
            position = cash / price
            cash = 0
            position_type = "short"
            entry_price = price
            trade_count += 1
            if predicted_sell_point is None:
                buy_idx = idx
        elif position_type == "long":
            if price < entry_price * (1 - stop_loss_pct):
                cash = position * price
                position = 0
                position_type = None
                trade_count += 1
                predicted_sell_point = idx
                predicted_sell_price = price
                entry_price = None
            elif pred == 0:
                cash = position * price
                position = 0
                position_type = None
                trade_count += 1
                predicted_sell_point = idx
                predicted_sell_price = price
                entry_price = None
        elif position_type == "short":
            if price > entry_price * (1 + stop_loss_pct):
                cash = position * (2 * entry_price - price)
                position = 0
                position_type = None
                trade_count += 1
                predicted_sell_point = idx
                predicted_sell_price = price
                entry_price = None
            elif pred == 1:
                cash = position * (2 * entry_price - price)
                position = 0
                position_type = None
                trade_count += 1
                predicted_sell_point = idx
                predicted_sell_price = price
                entry_price = None

    if position_type == "long" and position > 0:
        cash = position * prices.loc[X_test.index[-1]]
        trade_count += 1
        predicted_sell_point = X_test.index[-1]
        predicted_sell_price = prices.loc[X_test.index[-1]]
    elif position_type == "short" and position > 0:
        price = prices.loc[X_test.index[-1]]
        cash = position * (2 * entry_price - price)
        trade_count += 1
        predicted_sell_point = X_test.index[-1]
        predicted_sell_price = price

    model_profit = cash - initial_cash
    model_return_pct = (model_profit / initial_cash) * 100

    buy_strength, buy_strength_label = get_buy_strength(best_test_acc, model_return_pct)

    # Save plot of trades and indicators
    plot_trades_and_indicators(prices, list(X_test.index), best_y_pred, symbol, best_model_name, features)

    return {
        "symbol": symbol,
        "best_model": best_model_name,
        "test_accuracy": best_test_acc,
        "model_profit": model_profit,
        "model_return_pct": model_return_pct,
        "num_trades": trade_count,
        "buy_strength": buy_strength,
        "buy_strength_label": buy_strength_label,
        "predicted_sell_point": predicted_sell_point,
        "predicted_sell_price": predicted_sell_price,
        "trainer": trainer
    }

# --- Main script ---
df = pd.read_csv("data/nasdaq_historical_closes_2023_onward.csv", index_col="date", parse_dates=True)
symbols = [col for col in df.columns if col not in ("date",)]

user_input = input("Enter comma-separated list of stock symbols to analyze (or press c to skip and use all): ").strip()
if user_input.lower() != 'c' and user_input:
    user_symbols = [s.strip().upper() for s in user_input.split(",")]
    symbols = [s for s in symbols if s in user_symbols]
random.shuffle(symbols)

results = []
trainer = ModelTrainer()

listener_thread = threading.Thread(target=listen_for_q, daemon=True)
listener_thread.start()

with ThreadPoolExecutor(max_workers=8) as executor:
    future_to_symbol = {executor.submit(process_symbol, symbol, df, trainer): symbol for symbol in symbols}
    try:
        for future in as_completed(future_to_symbol):
            if stop_flag:
                print("Stopping as requested by user.")
                break
            result = future.result()
            if result:
                results.append(result)
    except KeyboardInterrupt:
        print("\nStopped by user (KeyboardInterrupt).")

if not results:
    print("No stocks were processed. Please check your data and filters.")
    sys.exit(0)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("model_return_pct", ascending=False)

print("\nBest Model Gain/Loss Table (Ranked by Model Return %):")
print(results_df[["symbol", "best_model", "test_accuracy", "model_profit", "model_return_pct", "num_trades", "buy_strength_label", "predicted_sell_point", "predicted_sell_price"]].round(3).to_string(index=False))

net_gain = results_df["model_profit"].sum()
print(f"\nNet gain/loss across all stocks (best model only): ${net_gain:.2f}")

losses = results_df[results_df["model_profit"] < 0]
if not losses.empty:
    print("\nSymbols where the best model incurred a loss:")
    print(losses[["symbol", "best_model", "model_profit", "model_return_pct", "num_trades"]].round(3).to_string(index=False))
else:
    print("\nNo symbols where the best model incurred a loss.")

def plot_results_table(results_df):
    display_df = results_df[["symbol", "best_model", "test_accuracy", "model_profit", "model_return_pct", "num_trades", "buy_strength_label", "predicted_sell_point", "predicted_sell_price"]].round(3)
    fig, ax = plt.subplots(figsize=(min(18, 2 + 2*len(display_df)), 0.7*len(display_df)+2))
    ax.axis('off')
    mpl_table = ax.table(cellText=display_df.values, colLabels=display_df.columns, loc='center', cellLoc='center')
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(12)
    mpl_table.auto_set_column_width(col=list(range(len(display_df.columns))))
    plt.title("Best Model Gain/Loss Table (Ranked by Model Return %)", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "summary_table.png"))
    plt.show()

if not results_df.empty:
    plot_results_table(results_df)

results_df.to_csv("best_model_vs_hold_zresults.csv", index=False)
print("Saved best_model_vs_hold_results.csv")

print("\n=== AI Model Buy Recommendations for Monday, July 21, 2025 ===")
buy_recommendations = []

last_trading_date = df.index.max()  # Should be the last date in your data

for result in results:
    symbol = result["symbol"]
    best_model_name = result["best_model"]
    if best_model_name is None:
        continue
    prices = fetch_price_data_from_csv(df, symbol, "2023-01-01", "2025-07-19")
    if last_trading_date not in prices.index or prices.loc[last_trading_date] < 1:
        continue

    # Compute features for the last trading date
    price_series = prices.loc[:last_trading_date]
    features = pd.DataFrame({symbol: price_series})
    features['ma5'] = price_series.rolling(5).mean()
    features['ma20'] = price_series.rolling(20).mean()
    features = add_technical_indicators(features, symbol)
    features = features.dropna()
    feature_cols = trainer.feature_columns.get(f"{symbol}_{best_model_name}")
    if not feature_cols:
        continue
    features = features[feature_cols].iloc[[-1]].fillna(0)

    model = trainer.models[symbol][best_model_name]
    scaler = trainer.scalers[symbol][best_model_name]
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    if pred == 1:
        buy_recommendations.append(symbol)

if buy_recommendations:
    print("The following stocks are recommended to BUY on Monday (next trading day after your data):")
    for s in buy_recommendations:
        print(f"  - {s}")
else:
    print("No stocks are recommended to buy on Monday by the models.")
