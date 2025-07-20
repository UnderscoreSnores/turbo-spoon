from models.train import ModelTrainer, add_technical_indicators
import pandas as pd

df = pd.read_csv("data/nasdaq_historical_closes_2023_onward.csv", index_col="date", parse_dates=True)
trainer = ModelTrainer()
for symbol in df.columns:
    features = pd.DataFrame({symbol: df[symbol]})
    features['ma5'] = df[symbol].rolling(5).mean()
    features['ma20'] = df[symbol].rolling(20).mean()
    features = add_technical_indicators(features, symbol)
    features = features.dropna()
    trainer.train_single_symbol(features, symbol, train_end_date="2024-01-01", multiclass=False)
    # Only save if a model was actually trained
    if symbol in trainer.models and trainer.models[symbol]:
        trainer.save_model(symbol)
    else:
        print(f"Skipping save for {symbol}: no model was trained.")
print("Model training complete.")
