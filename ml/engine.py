"""
ML/AI Trading Engine — the core intelligence layer.

This module provides:
  1. Ensemble model combining GARCH, LSTM, XGBoost, and LightGBM
  2. Feature engineering pipeline for market data
  3. Model calibration (Platt scaling / isotonic regression)
  4. Automated retraining on new resolved markets
  5. Sentiment analysis via transformer models
  6. Confidence estimation

The ensemble approach is key: no single model dominates across all market
conditions. GARCH captures volatility clustering, LSTM captures sequential
patterns, XGBoost/LightGBM capture non-linear feature interactions.
Sentiment models adjust for news-driven tail events.
"""
import json
import logging
import os
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss

from config.settings import settings

logger = logging.getLogger(__name__)

os.makedirs(settings.ml_model_dir, exist_ok=True)


class FeatureEngine:
    """Extract and engineer features from raw market data."""

    @staticmethod
    def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators from OHLCV data."""
        if df.empty or len(df) < 20:
            return df

        df = df.copy()

        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Volatility (rolling)
        for window in [5, 10, 20, 50]:
            df[f"volatility_{window}"] = df["returns"].rolling(window).std()

        # Moving averages
        for period in [7, 14, 20, 50]:
            df[f"sma_{period}"] = df["close"].rolling(period).mean()
            df[f"ema_{period}"] = df["close"].ewm(span=period).mean()

        # RSI
        for period in [14, 28]:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.finfo(float).eps)
            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        sma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_lower"] = sma20 - 2 * std20
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        ).replace(0, np.finfo(float).eps)

        # Average True Range (ATR)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()

        # Volume features
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(
            0, np.finfo(float).eps
        )
        df["vwap"] = (
            (df["close"] * df["volume"]).rolling(20).sum()
            / df["volume"].rolling(20).sum().replace(0, np.finfo(float).eps)
        )

        # Momentum
        for period in [5, 10, 20]:
            df[f"momentum_{period}"] = df["close"] / df["close"].shift(period) - 1

        # Price rate of change
        df["roc_10"] = df["close"].pct_change(10)
        df["roc_20"] = df["close"].pct_change(20)

        return df

    @staticmethod
    def compute_funding_features(funding_rates: pd.DataFrame) -> pd.DataFrame:
        """Compute features from funding rate data."""
        if funding_rates.empty or len(funding_rates) < 5:
            return funding_rates

        df = funding_rates.copy()

        # Funding rate stats
        for window in [8, 24, 72]:  # ~1 day, 3 days, 9 days (3x8h funding periods)
            df[f"fr_mean_{window}"] = df["funding_rate"].rolling(window).mean()
            df[f"fr_std_{window}"] = df["funding_rate"].rolling(window).std()
            df[f"fr_max_{window}"] = df["funding_rate"].rolling(window).max()
            df[f"fr_min_{window}"] = df["funding_rate"].rolling(window).min()

        # Cumulative funding
        df["fr_cumulative_24h"] = df["funding_rate"].rolling(8).sum()
        df["fr_cumulative_7d"] = df["funding_rate"].rolling(56).sum()

        # Rate of change
        df["fr_roc"] = df["funding_rate"].diff(3)

        # Positive/negative streaks
        df["fr_positive"] = (df["funding_rate"] > 0).astype(int)
        df["fr_streak"] = df["fr_positive"].groupby(
            (df["fr_positive"] != df["fr_positive"].shift()).cumsum()
        ).cumcount() + 1
        df.loc[df["fr_positive"] == 0, "fr_streak"] *= -1

        return df

    @staticmethod
    def compute_orderbook_features(orderbook: Dict) -> Dict:
        """Compute features from orderbook snapshot."""
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        features = {}
        if not bids or not asks:
            return features

        bid_sizes = [float(b.get("size", 0)) for b in bids[:20]]
        ask_sizes = [float(a.get("size", 0)) for a in asks[:20]]

        total_bid = sum(bid_sizes)
        total_ask = sum(ask_sizes)
        total = total_bid + total_ask

        features["bid_ask_ratio"] = total_bid / total_ask if total_ask > 0 else 1.0
        features["spread"] = float(asks[0].get("price", 0)) - float(bids[0].get("price", 0))
        features["mid_price"] = (float(asks[0].get("price", 0)) + float(bids[0].get("price", 0))) / 2
        features["imbalance"] = (total_bid - total_ask) / total if total > 0 else 0
        features["bid_depth_5"] = sum(bid_sizes[:5])
        features["ask_depth_5"] = sum(ask_sizes[:5])
        features["bid_depth_10"] = sum(bid_sizes[:10])
        features["ask_depth_10"] = sum(ask_sizes[:10])

        return features

    @staticmethod
    def compute_sentiment_features(texts: List[str], model=None) -> Dict:
        """
        Compute sentiment features from news/article texts.
        Uses a pre-trained transformer model for sentiment analysis.
        """
        if not texts:
            return {
                "sentiment_mean": 0.0,
                "sentiment_std": 0.0,
                "sentiment_min": 0.0,
                "sentiment_max": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 0.0,
                "n_articles": 0,
            }

        if model is None:
            try:
                from transformers import pipeline
                model = pipeline(
                    "sentiment-analysis",
                    model=settings.ml_sentiment_model,
                    top_k=None,
                    truncation=True,
                    max_length=512,
                )
            except Exception as e:
                logger.warning("Failed to load sentiment model: %s", e)
                return {
                    "sentiment_mean": 0.0, "sentiment_std": 0.0,
                    "sentiment_min": 0.0, "sentiment_max": 0.0,
                    "positive_ratio": 0.0, "negative_ratio": 0.0,
                    "neutral_ratio": 0.0, "n_articles": len(texts),
                }

        sentiments = []
        for text in texts[:50]:  # cap at 50 to avoid long inference times
            try:
                result = model(text[:512])
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        result = result[0]
                    scores = {r["label"].lower(): r["score"] for r in result}
                    compound = (
                        scores.get("positive", 0)
                        - scores.get("negative", 0)
                    )
                    sentiments.append(compound)
            except Exception:
                continue

        if not sentiments:
            return {
                "sentiment_mean": 0.0, "sentiment_std": 0.0,
                "sentiment_min": 0.0, "sentiment_max": 0.0,
                "positive_ratio": 0.0, "negative_ratio": 0.0,
                "neutral_ratio": 0.0, "n_articles": len(texts),
            }

        arr = np.array(sentiments)
        return {
            "sentiment_mean": float(np.mean(arr)),
            "sentiment_std": float(np.std(arr)) if len(arr) > 1 else 0.0,
            "sentiment_min": float(np.min(arr)),
            "sentiment_max": float(np.max(arr)),
            "positive_ratio": float(np.mean(arr > 0.1)),
            "negative_ratio": float(np.mean(arr < -0.1)),
            "neutral_ratio": float(np.mean(np.abs(arr) <= 0.1)),
            "n_articles": len(texts),
        }


class GARCHModel:
    """GARCH/EGARCH volatility forecasting model."""

    def __init__(self):
        self.model = None
        self.fitted = False

    def fit(self, returns: pd.Series, model_type: str = "EGARCH") -> None:
        """Fit a GARCH model on return series."""
        from arch import arch_model

        if len(returns) < 50:
            return

        returns_clean = returns.dropna()
        if len(returns_clean) < 50:
            return

        try:
            am = arch_model(
                returns_clean * 100,
                vol="Garch",
                p=1,
                q=1,
                mean="Constant",
                dist="t",
            )
            self.model = am.fit(disp="off", show_warning=False)
            self.fitted = True
        except Exception as e:
            logger.error("GARCH fit failed: %s", e)
            self.fitted = False

    def forecast(self, horizon: int = 1) -> Optional[Dict]:
        """Forecast volatility for the next `horizon` periods."""
        if not self.fitted or self.model is None:
            return None
        try:
            forecast = self.model.forecast(horizon=horizon)
            variance = forecast.variance.iloc[-1].values
            return {
                "volatility": float(np.sqrt(variance[-1])) / 100,
                "variance": float(variance[-1]) / 10000,
                "horizon": horizon,
            }
        except Exception as e:
            logger.error("GARCH forecast failed: %s", e)
            return None

    def get_params(self) -> Dict:
        if self.model is None:
            return {}
        return {
            "omega": float(self.model.params.get("omega", 0)),
            "alpha": float(self.model.params.get("alpha[1]", 0)),
            "beta": float(self.model.params.get("beta[1]", 0)),
            "log_likelihood": float(self.model.loglikelihood),
            "aic": float(self.model.aic),
        }


class LSTMModel:
    """LSTM model for sequential price pattern recognition."""

    def __init__(self, input_size: int = 30, hidden_size: int = 64, num_layers: int = 2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = None
        self.scaler = None
        self.fitted = False

    def _build_model(self):
        import torch
        import torch.nn as nn

        class LSTMNetwork(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.2,
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                out = self.fc(lstm_out[:, -1, :])
                return out

        return LSTMNetwork(self.input_size, self.hidden_size, self.num_layers)

    def prepare_sequences(
        self, features: np.ndarray, labels: np.ndarray, seq_length: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input."""
        if len(features) < seq_length + 1:
            return np.array([]), np.array([])

        X, y = [], []
        for i in range(len(features) - seq_length):
            X.append(features[i : i + seq_length])
            y.append(labels[i + seq_length])

        return np.array(X), np.array(y)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> None:
        """Train the LSTM model."""
        if len(X) < batch_size:
            return

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.model = self._build_model()
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

        self.fitted = True

    def predict(self, X: np.ndarray) -> Optional[float]:
        """Predict probability for a single sequence."""
        if not self.fitted or self.model is None:
            return None

        import torch

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).unsqueeze(0) if X.ndim == 2 else torch.FloatTensor([X])
            prediction = self.model(X_tensor)
            return float(prediction.item())

    def save(self, path: str) -> None:
        if self.model is not None:
            import torch
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
            }, path)

    def load(self, path: str) -> None:
        import torch
        checkpoint = torch.load(path, map_location="cpu")
        self.input_size = checkpoint["input_size"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.model = self._build_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.fitted = True


class GradientBoostingModels:
    """XGBoost and LightGBM models for feature-based probability estimation."""

    def __init__(self):
        self.xgb_model = None
        self.lgbm_model = None
        self.feature_names: List[str] = []
        self.fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        xgb_params: Optional[Dict] = None,
        lgbm_params: Optional[Dict] = None,
    ) -> Dict:
        """Train both XGBoost and LightGBM models."""
        import xgboost as xgb
        import lightgbm as lgbm
        from sklearn.model_selection import TimeSeriesSplit

        self.feature_names = list(X.columns)

        default_xgb = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 5,
            "early_stopping_rounds": 30,
        }
        default_lgbm = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 20,
            "verbose": -1,
        }

        if xgb_params:
            default_xgb.update(xgb_params)
        if lgbm_params:
            default_lgbm.update(lgbm_params)

        tscv = TimeSeriesSplit(n_splits=3)
        scores = {"xgb": [], "lgbm": []}

        # XGBoost
        try:
            es_rounds = default_xgb.pop("early_stopping_rounds", 30)
            n_est = default_xgb.pop("n_estimators", 300)
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=n_est, **default_xgb
            )
            train_idx, val_idx = list(tscv.split(X))[-1]
            self.xgb_model.fit(
                X.iloc[train_idx], y.iloc[train_idx],
                eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                verbose=False,
            )
            xgb_pred = self.xgb_model.predict_proba(X.iloc[val_idx])[:, 1]
            scores["xgb"] = brier_score_loss(y.iloc[val_idx], xgb_pred)
        except Exception as e:
            logger.error("XGBoost training failed: %s", e)

        # LightGBM
        try:
            n_est = default_lgbm.pop("n_estimators", 300)
            self.lgbm_model = lgbm.LGBMClassifier(
                n_estimators=n_est, **default_lgbm
            )
            self.lgbm_model.fit(
                X.iloc[train_idx], y.iloc[train_idx],
                eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            )
            lgbm_pred = self.lgbm_model.predict_proba(X.iloc[val_idx])[:, 1]
            scores["lgbm"] = brier_score_loss(y.iloc[val_idx], lgbm_pred)
        except Exception as e:
            logger.error("LightGBM training failed: %s", e)

        self.fitted = True
        return scores

    def predict(self, X: pd.DataFrame) -> Dict[str, float]:
        """Get predictions from both models."""
        result = {}
        if self.xgb_model is not None:
            try:
                result["xgb"] = float(self.xgb_model.predict_proba(X)[:, 1][0])
            except Exception:
                pass
        if self.lgbm_model is not None:
            try:
                result["lgbm"] = float(self.lgbm_model.predict_proba(X)[:, 1][0])
            except Exception:
                pass
        return result

    def get_feature_importance(self) -> Dict[str, float]:
        """Get combined feature importance from both models."""
        importance = {}
        if self.xgb_model is not None:
            xgb_imp = dict(zip(
                self.feature_names,
                self.xgb_model.feature_importances_,
            ))
            for k, v in xgb_imp.items():
                importance[f"xgb_{k}"] = float(v)
        if self.lgbm_model is not None:
            lgbm_imp = dict(zip(
                self.feature_names,
                self.lgbm_model.feature_importances_,
            ))
            for k, v in lgbm_imp.items():
                importance[f"lgbm_{k}"] = float(v)
        return importance

    def save(self, directory: str) -> None:
        import xgboost as xgb
        import lightgbm as lgbm
        if self.xgb_model is not None:
            self.xgb_model.save_model(os.path.join(directory, "xgb_model.json"))
        if self.lgbm_model is not None:
            self.lgbm_model.booster_.save_model(os.path.join(directory, "lgbm_model.txt"))
        with open(os.path.join(directory, "feature_names.json"), "w") as f:
            json.dump(self.feature_names, f)

    def load(self, directory: str) -> None:
        import xgboost as xgb
        import lightgbm as lgbm
        xgb_path = os.path.join(directory, "xgb_model.json")
        lgbm_path = os.path.join(directory, "lgbm_model.txt")
        if os.path.exists(xgb_path):
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(xgb_path)
        if os.path.exists(lgbm_path):
            self.lgbm_model = lgbm.LGBMClassifier()
            self.lgbm_model.booster_ = lgbm.Booster(model_file=lgbm_path)
        fn_path = os.path.join(directory, "feature_names.json")
        if os.path.exists(fn_path):
            with open(fn_path) as f:
                self.feature_names = json.load(f)
        self.fitted = self.xgb_model is not None or self.lgbm_model is not None


class ModelCalibrator:
    """Post-hoc probability calibration using Platt scaling or isotonic regression."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrator = None
        self.fitted = False

    def fit(self, probabilities: np.ndarray, true_labels: np.ndarray) -> None:
        """Fit calibrator on a held-out calibration set."""
        if len(probabilities) < 20:
            return

        if self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression(C=1.0)
            probs_2d = np.column_stack([1 - probabilities, probabilities])
            self.calibrator.fit(probs_2d, true_labels)
        else:  # isotonic
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(probabilities, true_labels)

        self.fitted = True

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply calibration to raw model probabilities."""
        if not self.fitted or self.calibrator is None:
            return probabilities

        if self.method == "platt":
            probs_2d = np.column_stack([1 - probabilities, probabilities])
            return self.calibrator.predict_proba(probs_2d)[:, 1]
        else:
            return self.calibrator.predict(probabilities)

    def evaluate(self, probabilities: np.ndarray, true_labels: np.ndarray) -> Dict:
        """Evaluate calibration quality."""
        if len(probabilities) < 10:
            return {"brier_score": None, "log_loss": None}

        return {
            "brier_score": brier_score_loss(true_labels, probabilities),
            "log_loss": log_loss(true_labels, np.clip(probabilities, 1e-7, 1 - 1e-7)),
            "mean_predicted": float(np.mean(probabilities)),
            "mean_actual": float(np.mean(true_labels)),
            "n_samples": len(probabilities),
        }


class EnsembleEngine:
    """
    The main ML/AI engine. Combines GARCH, LSTM, XGBoost, and LightGBM
    predictions using weighted ensemble, with calibrator applied to the
    final output.
    """

    def __init__(self):
        self.garch = GARCHModel()
        self.lstm = LSTMModel()
        self.gb_models = GradientBoostingModels()
        self.calibrator = ModelCalibrator(method="isotonic")
        self.feature_engine = FeatureEngine()
        self.sentiment_model = None
        self.weights = settings.ml_ensemble_weights  # [GARCH, LSTM, XGBoost, LightGBM]
        self.last_train_time: Optional[datetime] = None
        self.prediction_history: List[Dict] = []

    def train(
        self,
        ohlcv_data: pd.DataFrame,
        funding_data: Optional[pd.DataFrame] = None,
        labels: Optional[pd.Series] = None,
        news_texts: Optional[List[str]] = None,
    ) -> Dict:
        """
        Train the full ensemble. Returns training metrics.
        labels: binary Series where 1 = YES resolved, 0 = NO resolved
        """
        results = {}
        features_df = self.feature_engine.compute_technical_features(ohlcv_data)

        # GARCH
        if "log_returns" in features_df.columns:
            self.garch.fit(features_df["log_returns"].dropna())
            results["garch"] = self.garch.get_params()

        # LSTM
        if labels is not None and len(features_df) >= 50:
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            feature_array = features_df[numeric_cols].fillna(0).values
            label_array = labels.values if hasattr(labels, "values") else labels
            X_seq, y_seq = self.lstm.prepare_sequences(feature_array, label_array)
            if len(X_seq) > 0:
                self.lstm.fit(X_seq, y_seq, epochs=50)
                results["lstm"] = {"trained": True, "samples": len(X_seq)}

        # XGBoost + LightGBM
        if labels is not None and len(features_df) >= 30:
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            feature_df = features_df[numeric_cols].fillna(0)
            aligned_labels = labels.reindex(feature_df.index).dropna()
            feature_df = feature_df.loc[aligned_labels.index]
            if len(feature_df) >= 30:
                gb_scores = self.gb_models.fit(feature_df, aligned_labels)
                results["gradient_boosting"] = gb_scores

        # Calibration
        if labels is not None and len(self.prediction_history) >= 20:
            hist_probs = np.array([p["raw_prob"] for p in self.prediction_history[-200:]])
            hist_labels = np.array([p["actual"] for p in self.prediction_history[-200:] if "actual" in p])
            if len(hist_labels) == len(hist_probs) and len(hist_probs) >= 20:
                self.calibrator.fit(hist_probs, hist_labels)
                results["calibration"] = self.calibrator.evaluate(
                    hist_probs, hist_labels
                )

        self.last_train_time = datetime.utcnow()
        return results

    def predict(
        self,
        ohlcv_data: pd.DataFrame,
        funding_data: Optional[pd.DataFrame] = None,
        orderbook: Optional[Dict] = None,
        news_texts: Optional[List[str]] = None,
        market_price: float = 0.5,
        strike_price: Optional[float] = None,
        resolution_date: Optional[str] = None,
    ) -> Dict:
        """
        Generate ensemble prediction. Returns:
        {
            "probability": float,       # calibrated P(YES)
            "confidence": float,        # model confidence 0-1
            "raw_probabilities": dict,  # per-model probabilities
            "features_used": dict,      # feature summary
            "sentiment_adjustment": float,
        }
        """
        features_df = self.feature_engine.compute_technical_features(ohlcv_data)
        predictions = {}
        weights_used = {}

        # GARCH prediction
        if self.garch.fitted:
            garch_result = self.garch.forecast()
            if garch_result:
                vol = garch_result["volatility"]
                # Convert volatility forecast to probability using lognormal model
                if strike_price and market_price > 0:
                    import math
                    T = 1.0  # default 1 year, should be adjusted to resolution date
                    if resolution_date:
                        try:
                            res_dt = datetime.fromisoformat(resolution_date)
                            days_to_res = (res_dt - datetime.utcnow()).days
                            T = max(days_to_res / 365.25, 1 / 365.25)
                        except (ValueError, TypeError):
                            pass

                    mu = 0  # drift (could use historical mean return)
                    d1 = (math.log(market_price / strike_price) + (mu + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
                    from scipy.stats import norm
                    garch_prob = float(norm.cdf(d1))
                    predictions["garch"] = garch_prob
                    weights_used["garch"] = self.weights[0]

        # LSTM prediction
        if self.lstm.fitted and len(features_df) >= 30:
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            seq = features_df[numeric_cols].fillna(0).values[-30:]
            lstm_prob = self.lstm.predict(seq)
            if lstm_prob is not None:
                predictions["lstm"] = lstm_prob
                weights_used["lstm"] = self.weights[1]

        # XGBoost / LightGBM prediction
        if self.gb_models.fitted:
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            last_row = features_df[numeric_cols].fillna(0).iloc[[-1]]
            gb_preds = self.gb_models.predict(last_row)
            if "xgb" in gb_preds:
                predictions["xgb"] = gb_preds["xgb"]
                weights_used["xgb"] = self.weights[2]
            if "lgbm" in gb_preds:
                predictions["lgbm"] = gb_preds["lgbm"]
                weights_used["lgbm"] = self.weights[3]

        # Sentiment adjustment
        sentiment_adj = 0.0
        if news_texts and self.sentiment_model is None:
            try:
                from transformers import pipeline
                self.sentiment_model = pipeline(
                    "sentiment-analysis",
                    model=settings.ml_sentiment_model,
                    top_k=None,
                    truncation=True,
                    max_length=512,
                )
            except Exception as e:
                logger.warning("Failed to load sentiment model: %s", e)

        if news_texts and self.sentiment_model:
            sent_features = FeatureEngine.compute_sentiment_features(
                news_texts, self.sentiment_model
            )
            sentiment_adj = sent_features["sentiment_mean"] * 0.05  # max 5% adjustment

        # Weighted ensemble
        if predictions:
            total_weight = sum(weights_used.values())
            if total_weight > 0:
                raw_prob = sum(
                    predictions[k] * weights_used[k] for k in predictions
                ) / total_weight
            else:
                raw_prob = market_price
        else:
            raw_prob = market_price

        # Apply sentiment adjustment
        raw_prob = max(0.01, min(0.99, raw_prob + sentiment_adj))

        # Apply calibration
        raw_prob_arr = np.array([raw_prob])
        if self.calibrator.fitted:
            calibrated = self.calibrator.calibrate(raw_prob_arr)[0]
        else:
            calibrated = raw_prob

        calibrated = max(0.01, min(0.99, float(calibrated)))

        # Confidence estimation
        if len(predictions) > 1:
            pred_values = list(predictions.values())
            std = float(np.std(pred_values))
            confidence = max(0.0, 1.0 - std * 5)  # higher spread = lower confidence
        elif len(predictions) == 1:
            confidence = 0.5  # single model, moderate confidence
        else:
            confidence = 0.1  # no models available, low confidence

        # Orderbook adjustment
        if orderbook:
            ob_features = FeatureEngine.compute_orderbook_features(orderbook)
            if ob_features:
                imbalance = ob_features.get("imbalance", 0)
                confidence *= (1 + abs(imbalance) * 0.1)
                confidence = min(confidence, 1.0)

        result = {
            "probability": calibrated,
            "confidence": confidence,
            "raw_probabilities": predictions,
            "sentiment_adjustment": sentiment_adj,
            "weights_used": weights_used,
            "n_models": len(predictions),
        }

        self.prediction_history.append({
            "raw_prob": raw_prob,
            "calibrated_prob": calibrated,
            "confidence": confidence,
            "n_models": len(predictions),
            "timestamp": datetime.utcnow().isoformat(),
        })

        return result

    def should_retrain(self) -> bool:
        """Check if enough time has passed for retraining."""
        if self.last_train_time is None:
            return True
        elapsed = (datetime.utcnow() - self.last_train_time).total_seconds() / 3600
        return elapsed >= settings.ml_retrain_interval_hours

    def save_all(self, directory: Optional[str] = None) -> None:
        """Save all models to disk."""
        directory = directory or settings.ml_model_dir
        os.makedirs(directory, exist_ok=True)

        # GARCH params
        garch_path = os.path.join(directory, "garch_params.json")
        with open(garch_path, "w") as f:
            json.dump(self.garch.get_params(), f)

        # LSTM
        lstm_path = os.path.join(directory, "lstm_model.pt")
        self.lstm.save(lstm_path)

        # XGBoost + LightGBM
        self.gb_models.save(directory)

        # Calibrator
        cal_path = os.path.join(directory, "calibrator.pkl")
        with open(cal_path, "wb") as f:
            pickle.dump(self.calibrator, f)

        # Metadata
        meta = {
            "last_train_time": self.last_train_time.isoformat() if self.last_train_time else None,
            "weights": self.weights,
            "prediction_history_len": len(self.prediction_history),
        }
        with open(os.path.join(directory, "ensemble_meta.json"), "w") as f:
            json.dump(meta, f)

        logger.info("All models saved to %s", directory)

    def load_all(self, directory: Optional[str] = None) -> bool:
        """Load all models from disk. Returns True if any models were loaded."""
        directory = directory or settings.ml_model_dir
        loaded = False

        # LSTM
        lstm_path = os.path.join(directory, "lstm_model.pt")
        if os.path.exists(lstm_path):
            try:
                self.lstm.load(lstm_path)
                loaded = True
            except Exception as e:
                logger.warning("Failed to load LSTM: %s", e)

        # XGBoost + LightGBM
        gb_path = os.path.join(directory, "xgb_model.json")
        if os.path.exists(gb_path):
            try:
                self.gb_models.load(directory)
                loaded = True
            except Exception as e:
                logger.warning("Failed to load gradient boosting models: %s", e)

        # Calibrator
        cal_path = os.path.join(directory, "calibrator.pkl")
        if os.path.exists(cal_path):
            try:
                with open(cal_path, "rb") as f:
                    self.calibrator = pickle.load(f)
            except Exception as e:
                logger.warning("Failed to load calibrator: %s", e)

        if loaded:
            logger.info("Models loaded from %s", directory)
        return loaded
