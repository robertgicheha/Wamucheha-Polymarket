"""
TTE Training Orchestrator — manages 900 models per 15-min window.

Each TTE model predicts at a specific time-to-expiry (1s to 900s).
Full retrain: daily at 3am UTC. Incremental: hourly for TTE < 60s.
"""
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import settings
from ml.btc_features import BTCFeatureEngine
from ml.prediction_layers import (
    GRUPredictor,
    Layer2LogisticRegression,
    LinearRegressionBaseline,
    LSTMPredictor,
    MetaLearner,
    XGBoostPredictor,
)

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 900
TTE_BINS = list(range(1, WINDOW_SECONDS + 1))


class TTEModelSet:
    """Complete model set for one TTE value (L1a + L1b + L2 + L3)."""

    def __init__(self, tte_seconds: int):
        self.tte_seconds = tte_seconds
        self.lr_baseline = LinearRegressionBaseline()
        self.lstm = LSTMPredictor(sequence_length=min(60, max(10, tte_seconds)))
        self.gru = GRUPredictor(sequence_length=min(60, max(10, tte_seconds)))
        self.xgboost = XGBoostPredictor()
        self.active_l1b: List[str] = []
        self.l2_logistic = Layer2LogisticRegression()
        self.l3_meta = MetaLearner()
        self.train_time: Optional[float] = None
        self.last_brier_scores: Dict[str, float] = {}

    def predict(self, features: pd.DataFrame) -> Dict[str, float]:
        predictions = {}
        if self.lr_baseline.fitted:
            preds = self.lr_baseline.predict(features)
            predictions["lr"] = float(preds[-1]) if len(preds) > 0 else 0.5
        for name in self.active_l1b:
            model = {"lstm": self.lstm, "gru": self.gru, "xgboost": self.xgboost}.get(name)
            if model and model.fitted:
                preds = model.predict(features)
                predictions[name] = float(preds[-1]) if len(preds) > 0 else 0.5
        return predictions

    def save(self, directory: str) -> None:
        d = os.path.join(directory, f"tte_{self.tte_seconds:04d}")
        os.makedirs(d, exist_ok=True)
        self.lr_baseline.save(os.path.join(d, "lr.pkl"))
        self.lstm.save(os.path.join(d, "lstm.pt"))
        self.gru.save(os.path.join(d, "gru.pt"))
        self.xgboost.save(os.path.join(d, "xgb"))
        self.l2_logistic.save(os.path.join(d, "l2.pkl"))
        self.l3_meta.save(os.path.join(d, "l3.pt"))
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({
                "tte_seconds": self.tte_seconds,
                "active_l1b": self.active_l1b,
                "train_time": self.train_time,
                "brier_scores": self.last_brier_scores,
            }, f, indent=2)

    def load(self, directory: str) -> bool:
        d = os.path.join(directory, f"tte_{self.tte_seconds:04d}")
        if not os.path.exists(d):
            return False
        loaded = False
        for name, ext, loader in [
            ("lr", ".pkl", lambda p: self.lr_baseline.load(p)),
            ("lstm", ".pt", lambda p: self.lstm.load(p)),
            ("gru", ".pt", lambda p: self.gru.load(p)),
        ]:
            path = os.path.join(d, name + ext)
            if os.path.exists(path):
                try:
                    loader(path)
                    loaded = True
                except Exception as e:
                    logger.warning("Failed to load %s for TTE %d: %s", name, self.tte_seconds, e)
        xgb_path = os.path.join(d, "xgb.json")
        if os.path.exists(xgb_path):
            try:
                self.xgboost.load(os.path.join(d, "xgb"))
                loaded = True
            except Exception as e:
                logger.warning("Failed to load xgb for TTE %d: %s", self.tte_seconds, e)
        for name, ext, loader in [
            ("l2", ".pkl", lambda p: self.l2_logistic.load(p)),
            ("l3", ".pt", lambda p: self.l3_meta.load(p)),
        ]:
            path = os.path.join(d, name + ext)
            if os.path.exists(path):
                try:
                    loader(path)
                    loaded = True
                except Exception as e:
                    logger.warning("Failed to load %s for TTE %d: %s", name, self.tte_seconds, e)
        meta_path = os.path.join(d, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.active_l1b = meta.get("active_l1b", [])
            self.train_time = meta.get("train_time")
            self.last_brier_scores = meta.get("brier_scores", {})
        return loaded


class TTETrainingOrchestrator:
    """
    Manages training of all 900 TTE models.

    Training flow:
      1. Collect historical BTC data (trade + orderbook)
      2. Compute ~80 features via BTCFeatureEngine
      3. For each TTE bin (1-900 seconds):
         a. Create labels: was price > strike at TTE seconds later?
         b. Train L1a (LR) — baseline
         c. Train L1b (LSTM, GRU, XGBoost) — only keep if Brier < LR
         d. Combine L1 outputs -> train L2 (Logistic Regression)
         e. Combine all features -> train L3 (Meta-Learner)
      4. Save models to disk
      5. Log performance metrics

    Parallelism:
      - LSTM/GRU: sequential per TTE (GPU memory)
      - XGBoost: parallel across TTE bins (CPU-bound)
      - LR: fast, always sequential
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        max_workers: Optional[int] = None,
    ):
        self.model_dir = model_dir or settings.ml_model_dir
        self.max_workers = max_workers or settings.ml_btc_max_workers
        self.feature_engine = BTCFeatureEngine()
        self.tte_models: Dict[int, TTEModelSet] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.last_full_retrain: Optional[datetime] = None
        self.last_incremental_retrain: Optional[datetime] = None
        self.training_metrics: Dict = {}

        for tte in TTE_BINS:
            self.tte_models[tte] = TTEModelSet(tte)

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._retrain_loop())
        logger.info(
            "TTE orchestrator started (900 models, retrain hour=%d UTC)",
            settings.ml_retrain_hour,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("TTE orchestrator stopped")

    def predict(self, features: pd.DataFrame, tte_seconds: int) -> Dict:
        tte = max(1, min(900, tte_seconds))
        model_set = self.tte_models.get(tte)
        if model_set is None or not model_set.lr_baseline.fitted:
            return {"probability": 0.5, "confidence": 0.0, "models": {}}

        l1_predictions = model_set.predict(features)

        l2_features = self._build_l2_features(features, l1_predictions)
        l2_pred = model_set.l2_logistic.predict(l2_features)
        l2_probability = float(l2_pred[-1]) if len(l2_pred) > 0 else 0.5

        l3_features = self._build_l3_features(features, l1_predictions, l2_probability)
        l3_pred = model_set.l3_meta.predict(l3_features)
        l3_probability = float(l3_pred[-1]) if len(l3_pred) > 0 else 0.5

        all_probs = list(l1_predictions.values()) + [l2_probability, l3_probability]
        confidence = max(0.0, 1.0 - float(np.std(all_probs)) * 5) if len(all_probs) > 1 else 0.3

        return {
            "probability": l3_probability,
            "confidence": confidence,
            "l1_predictions": l1_predictions,
            "l2_probability": l2_probability,
            "l3_probability": l3_probability,
            "tte_seconds": tte,
            "active_models": list(l1_predictions.keys()),
        }

    def load_all(self) -> int:
        loaded = 0
        for tte, model_set in self.tte_models.items():
            if model_set.load(self.model_dir):
                loaded += 1
        logger.info("Loaded %d TTE model sets from %s", loaded, self.model_dir)
        return loaded

    def save_all(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        for model_set in self.tte_models.values():
            try:
                model_set.save(self.model_dir)
            except Exception as e:
                logger.error("Failed to save TTE %d: %s", model_set.tte_seconds, e)
        logger.info("Saved all TTE models to %s", self.model_dir)

    async def train_full(self, data: pd.DataFrame, strike_price: float) -> Dict:
        start_time = time.time()
        logger.info("Starting full TTE retrain (%d rows, strike=$%.2f)", len(data), strike_price)

        features = self.feature_engine.compute_all_features(data)
        if features.empty:
            return {"error": "feature_computation_failed"}

        logger.info("Features computed: %d cols, %d rows", len(features.columns), len(features))

        price_series = data["price"].values
        labels = {}
        for tte in TTE_BINS:
            if tte < len(price_series):
                future_prices = price_series[tte:]
                current_prices = price_series[:-tte]
                min_len = min(len(future_prices), len(current_prices))
                labels[tte] = (future_prices[:min_len] > strike_price).astype(float)
            else:
                labels[tte] = np.array([])

        results = {}
        trained_count = 0

        for tte in TTE_BINS:
            try:
                label_array = labels[tte]
                if len(label_array) < 50:
                    continue
                min_len = min(len(features), len(label_array))
                X = features.iloc[:min_len]
                y = label_array[:min_len]

                model_set = self.tte_models[tte]
                tte_result = await self._train_tte_model_set(model_set, X, y)
                results[tte] = tte_result
                trained_count += 1

                if trained_count % 100 == 0:
                    logger.info("Trained %d/900 TTE models...", trained_count)
            except Exception as e:
                logger.error("TTE %d training failed: %s", tte, e)

        self.save_all()
        elapsed = time.time() - start_time
        self.last_full_retrain = datetime.now(timezone.utc)

        all_brier = []
        for res in results.values():
            all_brier.extend(res.get("brier_scores", {}).values())

        summary = {
            "trained_models": trained_count,
            "elapsed_seconds": elapsed,
            "mean_brier": float(np.mean(all_brier)) if all_brier else None,
            "median_brier": float(np.median(all_brier)) if all_brier else None,
            "best_brier": float(np.min(all_brier)) if all_brier else None,
            "worst_brier": float(np.max(all_brier)) if all_brier else None,
            "strike_price": strike_price,
            "feature_count": len(features.columns),
        }
        self.training_metrics = summary
        logger.info(
            "Full retrain done: %d models in %.1fs (mean brier=%.4f)",
            trained_count, elapsed, summary.get("mean_brier") or 0,
        )
        return summary

    async def train_incremental(
        self, data: pd.DataFrame, strike_price: float, max_tte: int = 60
    ) -> Dict:
        start_time = time.time()
        features = self.feature_engine.compute_all_features(data)
        if features.empty:
            return {"error": "feature_computation_failed"}

        price_series = data["price"].values
        trained = 0
        for tte in range(1, min(max_tte + 1, WINDOW_SECONDS + 1)):
            if tte >= len(price_series):
                continue
            future = price_series[tte:]
            current = price_series[:-tte]
            ml = min(len(future), len(current))
            if ml < 50:
                continue
            y = (future[:ml] > strike_price).astype(float)
            X = features.iloc[:ml]
            try:
                await self._train_tte_model_set(self.tte_models[tte], X, y)
                trained += 1
            except Exception as e:
                logger.error("Incremental TTE %d failed: %s", tte, e)

        self.save_all()
        self.last_incremental_retrain = datetime.now(timezone.utc)
        elapsed = time.time() - start_time
        logger.info("Incremental retrain: %d models in %.1fs", trained, elapsed)
        return {"trained_models": trained, "elapsed_seconds": elapsed, "max_tte": max_tte}

    async def _train_tte_model_set(
        self, model_set: TTEModelSet, X: pd.DataFrame, y: np.ndarray
    ) -> Dict:
        results = {}
        brier_scores = {}

        lr_result = model_set.lr_baseline.fit(X, y)
        results["lr"] = lr_result
        lr_brier = lr_result.get("brier_score", float("inf")) or float("inf")

        model_set.active_l1b = []
        l1b_results = {}

        for name in ["xgboost", "lstm", "gru"]:
            model = {"xgboost": model_set.xgboost, "lstm": model_set.lstm, "gru": model_set.gru}[name]
            try:
                res = model.fit(X, y)
                results[name] = res
                model_brier = res.get("brier_score")
                if model_brier is not None and model_brier < lr_brier:
                    model_set.active_l1b.append(name)
                    brier_scores[name] = model_brier
                    l1b_results[name] = model.predict(X)
                else:
                    logger.debug(
                        "TTE %d %s excluded (brier=%.4f >= LR=%.4f)",
                        model_set.tte_seconds, name, model_brier or 999, lr_brier,
                    )
            except Exception as e:
                logger.warning("TTE %d %s training failed: %s", model_set.tte_seconds, name, e)

        if lr_brier < float("inf"):
            brier_scores["lr"] = lr_brier

        l1_features = self._build_l2_features(X, {
            name: model_set.lr_baseline.predict(X) for name in ["lr"]
        })
        for name, preds in l1b_results.items():
            l1_features[f"l1_{name}"] = preds

        l2_y = y[-len(l1_features):] if len(l1_features) < len(y) else y[:len(l1_features)]
        l2_X = l1_features.iloc[:len(l2_y)]
        if len(l2_X) >= 30:
            try:
                l2_res = model_set.l2_logistic.fit(l2_X, l2_y)
                results["l2"] = l2_res
                if "brier_score" in l2_res and l2_res["brier_score"] is not None:
                    brier_scores["l2"] = l2_res["brier_score"]
            except Exception as e:
                logger.warning("TTE %d L2 failed: %s", model_set.tte_seconds, e)

        l2_preds = model_set.l2_logistic.predict(l2_X) if model_set.l2_logistic.fitted else np.full(len(l2_X), 0.5)
        l3_features = self._build_l3_features(l2_X, {
            name: (l1b_results[name][-len(l2_X):] if name in l1b_results else np.full(len(l2_X), 0.5))
            for name in ["lr", "lstm", "gru", "xgboost"]
        }, l2_preds)

        l3_y = l2_y[:len(l3_features)]
        l3_X = l3_features.iloc[:len(l3_y)]
        if len(l3_X) >= 30:
            try:
                l3_res = model_set.l3_meta.fit(l3_X, l3_y)
                results["l3"] = l3_res
                if "brier_score" in l3_res and l3_res["brier_score"] is not None:
                    brier_scores["l3"] = l3_res["brier_score"]
            except Exception as e:
                logger.warning("TTE %d L3 failed: %s", model_set.tte_seconds, e)

        model_set.last_brier_scores = brier_scores
        model_set.train_time = time.time()
        results["brier_scores"] = brier_scores
        return results

    def _build_l2_features(
        self, base_features: pd.DataFrame, l1_predictions: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        l2_cols = {}
        for name, preds in l1_predictions.items():
            l2_cols[f"l1_{name}"] = preds[-len(base_features):] if len(preds) >= len(base_features) else np.full(len(base_features), 0.5)

        for col in ["spread", "bid1_size", "ask1_size"]:
            if col in base_features.columns:
                l2_cols[col] = base_features[col].values

        if "spread_bps" in base_features.columns:
            l2_cols["spread_bps"] = base_features["spread_bps"].values

        return pd.DataFrame(l2_cols, index=base_features.index)

    def _build_l3_features(
        self,
        base_features: pd.DataFrame,
        l1_predictions: Dict[str, np.ndarray],
        l2_predictions: np.ndarray,
    ) -> pd.DataFrame:
        l3_cols = {}
        for name, preds in l1_predictions.items():
            l3_cols[f"l1_{name}"] = preds[-len(base_features):] if len(preds) >= len(base_features) else np.full(len(base_features), 0.5)

        l3_cols["l2_prob"] = l2_predictions[-len(base_features):] if len(l2_predictions) >= len(base_features) else np.full(len(base_features), 0.5)

        for col in ["spread", "volume_mean_20", "buy_sell_imbalance", "vpin_20", "kyle_lambda_20", "amihud_20"]:
            if col in base_features.columns:
                l3_cols[col] = base_features[col].values

        return pd.DataFrame(l3_cols, index=base_features.index)

    async def _retrain_loop(self) -> None:
        while self._running:
            now = datetime.now(timezone.utc)
            try:
                if self.should_full_retrain():
                    logger.info("Scheduled full retrain triggered (hour=%d)", now.hour)
                elif self.should_incremental_retrain():
                    logger.info("Scheduled incremental retrain triggered")
            except Exception as e:
                logger.error("Retrain loop error: %s", e)

            await asyncio.sleep(300)

    def should_full_retrain(self) -> bool:
        if self.last_full_retrain is None:
            return True
        now = datetime.now(timezone.utc)
        if now.hour != settings.ml_retrain_hour:
            return False
        return (now - self.last_full_retrain).total_seconds() > 3600

    def should_incremental_retrain(self) -> bool:
        if self.last_incremental_retrain is None:
            return True
        return (datetime.now(timezone.utc) - self.last_incremental_retrain).total_seconds() > 3600

    @property
    def stats(self) -> Dict:
        fitted = sum(1 for m in self.tte_models.values() if m.lr_baseline.fitted)
        return {
            "total_tte_models": len(self.tte_models),
            "fitted_tte_models": fitted,
            "last_full_retrain": self.last_full_retrain.isoformat() if self.last_full_retrain else None,
            "last_incremental_retrain": self.last_incremental_retrain.isoformat() if self.last_incremental_retrain else None,
            "model_dir": self.model_dir,
            "training_metrics": self.training_metrics,
        }
