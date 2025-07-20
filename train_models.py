from models.train import ModelTrainer, add_technical_indicators
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

def train_and_save(symbol):
    trainer = ModelTrainer()
    df = pd.read_csv("data/nasdaq_historical_closes_2023_onward.csv", index_col="date", parse_dates=True)
    features = pd.DataFrame({symbol: df[symbol]})
    features['ma5'] = df[symbol].rolling(5).mean()
    features['ma20'] = df[symbol].rolling(20).mean()
    features = add_technical_indicators(features, symbol)
    features = features.dropna()
    trainer.train_single_symbol(features, symbol, train_end_date="2024-01-01", multiclass=False)
    if symbol in trainer.models and trainer.models[symbol]:
        trainer.save_model(symbol)
        return (symbol, True)
    else:
        return (symbol, False)

if __name__ == "__main__":
    df = pd.read_csv("data/nasdaq_historical_closes_2023_onward.csv", index_col="date", parse_dates=True)
    symbols = list(df.columns)
    trained_count = 0
    skipped_count = 0
    total = len(symbols)

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(train_and_save, symbol): symbol for symbol in symbols}
        for idx, future in enumerate(as_completed(futures), 1):
            symbol, trained = future.result()
            if trained:
                trained_count += 1
                print(f"[{idx}/{total}] ✔ Model trained and saved for {symbol} ({trained_count} so far)")
            else:
                skipped_count += 1
                print(f"[{idx}/{total}] ✖ Skipped {symbol}: no model was trained ({skipped_count} skipped)")

    print(f"\nModel training complete. {trained_count} models trained, {skipped_count} skipped.")
