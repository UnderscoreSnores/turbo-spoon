from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import os

from models.train import ModelTrainer, compute_rsi, add_technical_indicators
from models.plot_utils import plot_interactive

# === CONFIGURATION ===
ALLOWED_ORIGINS = ["*"]  # Or set to your frontend domain(s)
DATA_PATH = "data/nasdaq_historical_closes_2023_onward.csv"

# === APP SETUP ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === LOAD DATA AND MODELS ON STARTUP ===
df = pd.read_csv(DATA_PATH, index_col="date", parse_dates=True)
trainer = ModelTrainer()
for symbol in df.columns:
    try:
        trainer.load_model(symbol)
    except Exception:
        pass

class SymbolsRequest(BaseModel):
    symbols: List[str]

def get_features_for_symbol(symbol, date=None):
    prices = df[symbol]
    if date is not None:
        prices = prices.loc[:date]
    features = pd.DataFrame({symbol: prices})
    features['ma5'] = prices.rolling(5).mean()
    features['ma20'] = prices.rolling(20).mean()
    features = add_technical_indicators(features, symbol)
    features = features.dropna()
    return features

@app.get("/predict")
def predict(symbol: str):
    if symbol not in trainer.models or not trainer.models[symbol]:
        return {"error": f"No model for {symbol}"}
    best_model_name = list(trainer.models[symbol].keys())[0]
    model = trainer.models[symbol][best_model_name]
    scaler = trainer.scalers[symbol][best_model_name]
    feature_cols = trainer.feature_columns[f"{symbol}_{best_model_name}"]
    features = get_features_for_symbol(symbol)
    if features.empty:
        return {"error": "No features for symbol"}
    features = features[feature_cols].iloc[[-1]].fillna(0)
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    return {"symbol": symbol, "prediction": int(pred)}

@app.post("/batch_predict")
def batch_predict(request: SymbolsRequest):
    results = []
    for symbol in request.symbols:
        try:
            res = predict(symbol)
            results.append(res)
        except Exception as e:
            results.append({"symbol": symbol, "error": str(e)})
    return results

@app.get("/recommendations")
def recommendations():
    last_trading_date = df.index.max()
    buy_recommendations = []
    for symbol in df.columns:
        if symbol not in trainer.models or not trainer.models[symbol]:
            continue
        best_model_name = list(trainer.models[symbol].keys())[0]
        model = trainer.models[symbol][best_model_name]
        scaler = trainer.scalers[symbol][best_model_name]
        feature_cols = trainer.feature_columns[f"{symbol}_{best_model_name}"]
        prices = df[symbol].loc[:last_trading_date]
        features = pd.DataFrame({symbol: prices})
        features['ma5'] = prices.rolling(5).mean()
        features['ma20'] = prices.rolling(20).mean()
        features = add_technical_indicators(features, symbol)
        features = features.dropna()
        if features.empty or not all(col in features.columns for col in feature_cols):
            continue
        features = features[feature_cols].iloc[[-1]].fillna(0)
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        if pred == 1:
            buy_recommendations.append(symbol)
    return {"buy": buy_recommendations}

@app.get("/plot_interactive")
def plot_interactive_endpoint(symbol: str):
    prices = df[symbol].tail(100)
    features = pd.DataFrame({symbol: prices})
    features['ma5'] = prices.rolling(5).mean()
    features['ma20'] = prices.rolling(20).mean()
    features = add_technical_indicators(features, symbol)
    features = features.dropna()
    html = plot_interactive(prices, features, symbol)
    return Response(content=html, media_type="text/html")
