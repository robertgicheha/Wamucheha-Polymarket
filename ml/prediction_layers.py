"""
Multi-layer BTC prediction pipeline.

Layer 1a — Linear Regression baseline (gold standard).
Layer 1b — LSTM, GRU, XGBoost (must beat LR to be included).
Layer 2  — Logistic Regression on microstructure + L1 outputs.
Layer 3  — Neural network meta-learner on all features for final edge.

Each layer is trained independently and produces probability outputs.
The meta-learner (Layer 3) combines all layer outputs + raw features.

Constraint: Every L1b model MUST outperform the LR baseline on Brier score.
If it doesn't, it's excluded from the ensemble.
"""
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, mean_squared_error
from sklearn.preprocessing import StandardScaler

from config.settings import settings

logger = logging.getLogger(__name__)


# ── Layer 1a: Linear Regression Baseline ──────────────────────────────

class LinearRegressionBaseline:
    """
    Gold standard baseline. Every L1b model must beat this.

    Predicts the probability that BTC will be above the strike price
    at the given time-to-expiry, using linear regression on features.

    Simple, fast, interpretable — and hard to beat consistently.
    """

    def __init__(self):
        self.model: Optional[LinearRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.brier_score: Optional[float] = None
        self.feature_names: List[str] = []
        self.fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        validation_split: float = 0.2,
    ) -> Dict:
        """Train and evaluate the linear regression baseline."""
        if len(X) < 50:
            logger.warning("LR baseline: insufficient data (%d rows)", len(X))
            return {"error": "insufficient_data"}

        self.feature_names = list(X.columns)

        # Split chronologically
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Fit
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_val_scaled)
        y_pred_clipped = np.clip(y_pred, 0.01, 0.99)

        self.brier_score = brier_score_loss(y_val, y_pred_clipped)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_clipped)))

        self.fitted = True

        # Feature importance (absolute coefficients)
        importance = dict(zip(
            self.feature_names,
            np.abs(self.model.coef_).tolist(),
        ))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        results = {
            "brier_score": self.brier_score,
            "rmse": rmse,
            "r_squared": float(self.model.score(X_val_scaled, y_val)),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "top_features": top_features,
        }

        logger.info(
            "LR baseline: brier=%.4f, rmse=%.4f, r²=%.4f",
            self.brier_score, rmse, results["r_squared"],
        )

        return results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities from features."""
        if not self.fitted or self.model is None or self.scaler is None:
            return np.full(len(X), 0.5)

        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return np.clip(preds, 0.01, 0.99)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "brier_score": self.brier_score,
                "feature_names": self.feature_names,
            }, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.brier_score = data["brier_score"]
        self.feature_names = data["feature_names"]
        self.fitted = True


# ── Layer 1b: Advanced Models ─────────────────────────────────────────

class LSTMPredictor:
    """
    LSTM model for sequential BTC price prediction.

    Takes sequences of feature vectors and predicts probability
    of price being above strike at time-to-expiry.
    """

    def __init__(
        self,
        input_size: int = 80,
        hidden_size: int = 128,
        num_layers: int = 2,
        sequence_length: int = 60,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.brier_score: Optional[float] = None
        self.fitted = False

    def _build_model(self):
        import torch
        import torch.nn as nn

        class LSTMNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.3,
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.fc(lstm_out[:, -1, :])

        return LSTMNet(self.input_size, self.hidden_size, self.num_layers)

    def _make_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(X) < self.sequence_length + 1:
            return np.array([]), np.array([])
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i : i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        epochs: Optional[int] = None,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ) -> Dict:
        """Train LSTM and evaluate against LR baseline."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        epochs = epochs or settings.ml_lstm_epochs

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train_arr, X_val_arr = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train_arr, y_val_arr = y[:split_idx], y[split_idx:]

        X_train_seq, y_train_seq = self._make_sequences(X_train_arr, y_train_arr)
        X_val_seq, y_val_seq = self._make_sequences(X_val_arr, y_val_arr)

        if len(X_train_seq) < batch_size:
            return {"error": "insufficient_sequences"}

        self.input_size = X_train_seq.shape[2]
        self.model = self._build_model()

        X_tensor = torch.FloatTensor(X_train_seq)
        y_tensor = torch.FloatTensor(y_train_seq).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.BCELoss()

        self.model.train()
        best_val_brier = float("inf")
        patience_counter = 0
        max_patience = 15

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            # Validate
            if len(X_val_seq) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(torch.FloatTensor(X_val_seq)).numpy().flatten()
                val_brier = brier_score_loss(y_val_seq, np.clip(val_pred, 0.01, 0.99))
                scheduler.step(val_brier)
                self.model.train()

                if val_brier < best_val_brier:
                    best_val_brier = val_brier
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        break

        # Load best weights
        if "best_state" in locals():
            self.model.load_state_dict(best_state)

        # Final evaluation
        self.model.eval()
        if len(X_val_seq) > 0:
            with torch.no_grad():
                val_pred = self.model(torch.FloatTensor(X_val_seq)).numpy().flatten()
            self.brier_score = brier_score_loss(y_val_seq, np.clip(val_pred, 0.01, 0.99))
        else:
            self.brier_score = None

        self.fitted = True

        return {
            "brier_score": self.brier_score,
            "n_train_sequences": len(X_train_seq),
            "n_val_sequences": len(X_val_seq),
            "epochs_trained": epoch + 1,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict from latest sequence."""
        if not self.fitted or self.model is None or self.scaler is None:
            return np.full(len(X), 0.5)

        import torch

        X_scaled = self.scaler.transform(X)
        if len(X_scaled) < self.sequence_length:
            return np.full(len(X), 0.5)

        # Use last sequence_length rows
        seq = X_scaled[-self.sequence_length:]
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.FloatTensor(seq).unsqueeze(0)).item()
        return np.full(len(X), pred)

    def save(self, path: str) -> None:
        import torch
        torch.save({
            "model_state": self.model.state_dict() if self.model else None,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "sequence_length": self.sequence_length,
            "scaler": self.scaler,
            "brier_score": self.brier_score,
        }, path)

    def load(self, path: str) -> None:
        import torch
        data = torch.load(path, map_location="cpu")
        self.input_size = data["input_size"]
        self.hidden_size = data["hidden_size"]
        self.num_layers = data["num_layers"]
        self.sequence_length = data["sequence_length"]
        self.scaler = data["scaler"]
        self.brier_score = data["brier_score"]
        if data["model_state"] is not None:
            self.model = self._build_model()
            self.model.load_state_dict(data["model_state"])
            self.model.eval()
            self.fitted = True


class GRUPredictor:
    """
    GRU model — similar to LSTM but with gated recurrent units.
    Often faster to train, comparable performance on short sequences.
    """

    def __init__(
        self,
        input_size: int = 80,
        hidden_size: int = 128,
        num_layers: int = 2,
        sequence_length: int = 60,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.brier_score: Optional[float] = None
        self.fitted = False

    def _build_model(self):
        import torch
        import torch.nn as nn

        class GRUNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super().__init__()
                self.gru = nn.GRU(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.3,
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                gru_out, _ = self.gru(x)
                return self.fc(gru_out[:, -1, :])

        return GRUNet(self.input_size, self.hidden_size, self.num_layers)

    def _make_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(X) < self.sequence_length + 1:
            return np.array([]), np.array([])
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i : i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        epochs: Optional[int] = None,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ) -> Dict:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        epochs = epochs or settings.ml_lstm_epochs

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train_arr, X_val_arr = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train_arr, y_val_arr = y[:split_idx], y[split_idx:]

        X_train_seq, y_train_seq = self._make_sequences(X_train_arr, y_train_arr)
        X_val_seq, y_val_seq = self._make_sequences(X_val_arr, y_val_arr)

        if len(X_train_seq) < batch_size:
            return {"error": "insufficient_sequences"}

        self.input_size = X_train_seq.shape[2]
        self.model = self._build_model()

        X_tensor = torch.FloatTensor(X_train_seq)
        y_tensor = torch.FloatTensor(y_train_seq).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.BCELoss()

        self.model.train()
        best_val_brier = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            if len(X_val_seq) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(torch.FloatTensor(X_val_seq)).numpy().flatten()
                val_brier = brier_score_loss(y_val_seq, np.clip(val_pred, 0.01, 0.99))
                scheduler.step(val_brier)
                self.model.train()

                if val_brier < best_val_brier:
                    best_val_brier = val_brier
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= 15:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        if len(X_val_seq) > 0:
            with torch.no_grad():
                val_pred = self.model(torch.FloatTensor(X_val_seq)).numpy().flatten()
            self.brier_score = brier_score_loss(y_val_seq, np.clip(val_pred, 0.01, 0.99))
        else:
            self.brier_score = None

        self.fitted = True
        return {
            "brier_score": self.brier_score,
            "n_train_sequences": len(X_train_seq),
            "n_val_sequences": len(X_val_seq),
            "epochs_trained": epoch + 1,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted or self.model is None or self.scaler is None:
            return np.full(len(X), 0.5)

        import torch
        X_scaled = self.scaler.transform(X)
        if len(X_scaled) < self.sequence_length:
            return np.full(len(X), 0.5)

        seq = X_scaled[-self.sequence_length:]
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.FloatTensor(seq).unsqueeze(0)).item()
        return np.full(len(X), pred)

    def save(self, path: str) -> None:
        import torch
        torch.save({
            "model_state": self.model.state_dict() if self.model else None,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "sequence_length": self.sequence_length,
            "scaler": self.scaler,
            "brier_score": self.brier_score,
        }, path)

    def load(self, path: str) -> None:
        import torch
        data = torch.load(path, map_location="cpu")
        self.input_size = data["input_size"]
        self.hidden_size = data["hidden_size"]
        self.num_layers = data["num_layers"]
        self.sequence_length = data["sequence_length"]
        self.scaler = data["scaler"]
        self.brier_score = data["brier_score"]
        if data["model_state"] is not None:
            self.model = self._build_model()
            self.model.load_state_dict(data["model_state"])
            self.model.eval()
            self.fitted = True


class XGBoostPredictor:
    """
    XGBoost for tabular BTC prediction.
    Often the strongest performer on structured/tabular features.
    """

    def __init__(self):
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.brier_score: Optional[float] = None
        self.feature_names: List[str] = []
        self.fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        validation_split: float = 0.2,
    ) -> Dict:
        import xgboost as xgb

        self.feature_names = list(X.columns)

        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=self.feature_names,
            index=X_train.index,
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=self.feature_names,
            index=X_val.index,
        )

        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=30,
            verbosity=0,
        )

        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False,
        )

        y_pred = self.model.predict_proba(X_val_scaled)[:, 1]
        self.brier_score = brier_score_loss(y_val, np.clip(y_pred, 0.01, 0.99))
        self.fitted = True

        importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_.tolist(),
        ))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "brier_score": self.brier_score,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "best_iteration": self.model.best_iteration,
            "top_features": top_features,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted or self.model is None or self.scaler is None:
            return np.full(len(X), 0.5)

        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_names,
            index=X.index,
        )
        preds = self.model.predict_proba(X_scaled)[:, 1]
        return np.clip(preds, 0.01, 0.99)

    def save(self, path: str) -> None:
        import xgboost as xgb
        self.model.save_model(path + ".json")
        with open(path + ".pkl", "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "brier_score": self.brier_score,
                "feature_names": self.feature_names,
            }, f)

    def load(self, path: str) -> None:
        import xgboost as xgb
        self.model = xgb.XGBClassifier()
        self.model.load_model(path + ".json")
        with open(path + ".pkl", "rb") as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self.brier_score = data["brier_score"]
        self.feature_names = data["feature_names"]
        self.fitted = True


# ── Layer 2: Logistic Regression (microstructure + L1 outputs) ────────

class Layer2LogisticRegression:
    """
    Logistic Regression on top of:
    - Microstructure features (VPIN, Kyle's Lambda, Amihud, etc.)
    - Layer 1 probability outputs (LR, LSTM, GRU, XGBoost)

    This layer learns how to combine L1 predictions with
    microstructure signals for a more informed probability.
    """

    def __init__(self):
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.brier_score: Optional[float] = None
        self.feature_names: List[str] = []
        self.fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        validation_split: float = 0.2,
    ) -> Dict:
        if len(X) < 50:
            return {"error": "insufficient_data"}

        self.feature_names = list(X.columns)

        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        self.model = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
        )
        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict_proba(X_val_scaled)[:, 1]
        self.brier_score = brier_score_loss(y_val, np.clip(y_pred, 0.01, 0.99))
        self.fitted = True

        importance = dict(zip(
            self.feature_names,
            np.abs(self.model.coef_[0]).tolist(),
        ))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "brier_score": self.brier_score,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "top_features": top_features,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted or self.model is None or self.scaler is None:
            return np.full(len(X), 0.5)

        X_scaled = self.scaler.transform(X)
        preds = self.model.predict_proba(X_scaled)[:, 1]
        return np.clip(preds, 0.01, 0.99)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "brier_score": self.brier_score,
                "feature_names": self.feature_names,
            }, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.brier_score = data["brier_score"]
        self.feature_names = data["feature_names"]
        self.fitted = True


# ── Layer 3: Neural Network Meta-Learner ──────────────────────────────

class MetaLearner:
    """
    Neural network meta-learner — combines ALL layer outputs + raw features.

    Input features:
    - L1a probability (LR)
    - L1b probabilities (LSTM, GRU, XGBoost)
    - L2 probability (Logistic Regression)
    - Key microstructure features (VPIN, Kyle's Lambda, Amihud, spread, depth)
    - Time-to-expiry
    - Volatility features

    Output: Final calibrated probability (the edge signal)
    """

    def __init__(self, input_size: int = 30):
        self.input_size = input_size
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.brier_score: Optional[float] = None
        self.feature_names: List[str] = []
        self.fitted = False

    def _build_model(self):
        import torch
        import torch.nn as nn

        class MetaNet(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.net(x)

        return MetaNet(self.input_size)

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ) -> Dict:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.feature_names = list(X.columns)
        self.input_size = len(self.feature_names)

        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        self.model = self._build_model()

        X_tensor = torch.FloatTensor(X_train_scaled)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCELoss()

        self.model.train()
        best_val_brier = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # Validate
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(torch.FloatTensor(X_val_scaled)).numpy().flatten()
            val_brier = brier_score_loss(y_val, np.clip(val_pred, 0.01, 0.99))
            self.model.train()

            if val_brier < best_val_brier:
                best_val_brier = val_brier
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        with torch.no_grad():
            val_pred = self.model(torch.FloatTensor(X_val_scaled)).numpy().flatten()
        self.brier_score = brier_score_loss(y_val, np.clip(val_pred, 0.01, 0.99))
        self.fitted = True

        return {
            "brier_score": self.brier_score,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "epochs_trained": epoch + 1,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted or self.model is None or self.scaler is None:
            return np.full(len(X), 0.5)

        import torch

        X_scaled = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.FloatTensor(X_scaled)).numpy().flatten()
        return np.clip(preds, 0.01, 0.99)

    def save(self, path: str) -> None:
        import torch
        torch.save({
            "model_state": self.model.state_dict() if self.model else None,
            "input_size": self.input_size,
            "scaler": self.scaler,
            "brier_score": self.brier_score,
            "feature_names": self.feature_names,
        }, path)

    def load(self, path: str) -> None:
        import torch
        data = torch.load(path, map_location="cpu")
        self.input_size = data["input_size"]
        self.scaler = data["scaler"]
        self.brier_score = data["brier_score"]
        self.feature_names = data["feature_names"]
        if data["model_state"] is not None:
            self.model = self._build_model()
            self.model.load_state_dict(data["model_state"])
            self.model.eval()
            self.fitted = True
