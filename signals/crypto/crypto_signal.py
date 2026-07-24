"""
Crypto price-market signal generator.

Pipeline:
  1. Parse market question → extract asset, strike price, resolution date
  2. Pull multi-timeframe OHLCV from Binance/OKX
  3. Pull funding rate history from Binance/OKX
  4. Run GARCH → volatility forecast to resolution date
  5. Run LSTM → short-horizon directional momentum signal
  6. XGBoost/LightGBM → non-linear feature interactions
  7. Combine into calibrated P(YES) using ensemble engine
  8. Apply sentiment adjustment from news for tail events
  9. Cross-validate funding rates between exchanges for signal strength

This is the primary category — it has the most tractable ground truth.
"""
import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config.settings import settings
from data.market_parser import parse_crypto_market
from data.price_feeds import DataAggregator
from ml.engine import EnsembleEngine, FeatureEngine
from signals.base_signal import BaseSignalGenerator, SignalOutput

logger = logging.getLogger(__name__)


class CryptoSignalGenerator(BaseSignalGenerator):
    category = "crypto"

    def __init__(self):
        self.data = DataAggregator()
        self.ml_engine = EnsembleEngine()
        self.feature_engine = FeatureEngine()
        self.sentiment_model = None

        # Load pre-trained models if available
        self.ml_engine.load_all(settings.ml_model_dir)

    def generate(self, market_id: str, market_question: str = "", market_price: float = 0.5) -> SignalOutput:
        """
        Generate a calibrated P(YES) for a crypto price market.

        Args:
            market_id: Polymarket condition ID
            market_question: the market question text (e.g. "Will BTC be above $120k by Dec 31?")
            market_price: current market price for YES (0-1)
        """
        parsed = parse_crypto_market(market_question)
        asset = parsed["asset"]
        direction = parsed["direction"]
        strike = parsed["strike_price"]
        resolution_date = parsed["resolution_date"]

        if asset == "unknown" or strike == 0.0:
            return SignalOutput(
                market_id=market_id,
                model_probability=market_price,
                confidence=0.1,
                reasoning=f"Cannot parse market: {market_question}",
            )

        # Fetch data
        ohlcv_1h = self.data.get_ohlcv(asset, "1h", limit=200)
        ohlcv_15m = self.data.get_ohlcv(asset, "15m", limit=100)
        funding_df = self.data.get_funding_rates(asset, limit=100)

        if ohlcv_1h.empty:
            return SignalOutput(
                market_id=market_id,
                model_probability=market_price,
                confidence=0.1,
                reasoning=f"No OHLCV data available for {asset}",
            )

        # Get current price for reference
        ticker = self.data.get_ticker(asset)
        current_price = 0
        if ticker.get("binance"):
            current_price = ticker["binance"].get("last", 0)
        elif ticker.get("okx"):
            current_price = ticker["okx"].get("last", 0)

        # Get orderbook for depth signal
        orderbook = None
        try:
            from connectors.polymarket_connector import PolymarketConnector
            poly = PolymarketConnector()
            # Would need token_id for the specific market
        except Exception:
            pass

        # Get news for sentiment
        news_texts = self.data.get_headlines_for_market(market_question)

        # Run ML ensemble
        ml_result = self.ml_engine.predict(
            ohlcv_data=ohlcv_1h,
            funding_data=funding_df if not funding_df.empty else None,
            orderbook=orderbook,
            news_texts=news_texts,
            market_price=current_price,
            strike_price=strike,
            resolution_date=resolution_date.isoformat() if resolution_date else None,
        )

        # Adjust for direction
        raw_prob = ml_result["probability"]
        if direction == "below":
            raw_prob = 1.0 - raw_prob

        # Funding rate confirmation signal
        funding_signal = self._compute_funding_signal(funding_df)

        # Cross-exchange price agreement
        cross_exchange_signal = 0.0
        if ticker.get("price_disagreement_bps") is not None:
            disagreement_bps = ticker["price_disagreement_bps"]
            # High disagreement = lower confidence
            if disagreement_bps < 5:  # < 5 bps = good agreement
                cross_exchange_signal = 0.01
            elif disagreement_bps > 50:  # > 50 bps = significant disagreement
                cross_exchange_signal = -0.01

        # Multi-timeframe confirmation
        mtf_signal = self._compute_mtf_signal(asset)

        # Combine all signals
        final_prob = raw_prob + funding_signal * 0.02 + cross_exchange_signal + mtf_signal * 0.01
        final_prob = max(0.01, min(0.99, final_prob))

        # Confidence
        confidence = ml_result["confidence"]
        confidence *= self._confidence_adjustment(
            ohlcv_1h, funding_df, ticker, news_texts
        )

        # Build reasoning
        reasoning_parts = [
            f"Asset: {asset.upper()}, Direction: {direction}, Strike: ${strike:,.0f}",
            f"Ensemble ({ml_result['n_models']} models): {raw_prob:.3f}",
            f"Funding signal: {funding_signal:+.3f}",
            f"Cross-exchange signal: {cross_exchange_signal:+.3f}",
            f"MTF signal: {mtf_signal:+.3f}",
            f"Final: {final_prob:.3f} vs market: {market_price:.3f}",
            f"Confidence: {confidence:.2f}",
        ]
        if ml_result["sentiment_adjustment"] != 0:
            reasoning_parts.append(
                f"Sentiment adj: {ml_result['sentiment_adjustment']:+.3f}"
            )

        if settings.log_level == "DEBUG":
            reasoning_parts.append(
                f"Raw model probs: {ml_result['raw_probabilities']}"
            )

        return SignalOutput(
            market_id=market_id,
            model_probability=final_prob,
            confidence=confidence,
            reasoning=" | ".join(reasoning_parts),
        )

    def _compute_funding_signal(self, funding_df: pd.DataFrame) -> float:
        """
        Compute a signal from funding rates.
        Extreme positive funding = market is long-heavy → slight bearish signal
        Extreme negative funding = market is short-heavy → slight bullish signal
        """
        if funding_df.empty or len(funding_df) < 5:
            return 0.0

        recent_rates = funding_df["funding_rate"].tail(8)  # last ~1 day
        avg_rate = recent_rates.mean()
        cumulative = recent_rates.sum()

        # Normalize: funding rates are typically in the range -0.001 to 0.001
        signal = -np.clip(avg_rate * 100, -0.5, 0.5)

        # Strong cumulative funding in one direction adds conviction
        if abs(cumulative) > 0.005:
            signal *= 1.5

        return float(signal)

    def _compute_mtf_signal(self, asset: str) -> float:
        """
        Multi-timeframe momentum signal.
        Check if shorter timeframes confirm the longer-term trend.
        """
        try:
            df_15m = self.data.get_ohlcv(asset, "15m", limit=50)
            df_4h = self.data.get_ohlcv(asset, "4h", limit=50)
            df_1d = self.data.get_ohlcv(asset, "1d", limit=30)

            signals = []
            for df in [df_15m, df_4h, df_1d]:
                if df.empty or len(df) < 20:
                    continue
                closes = df["close"].values
                sma_short = np.mean(closes[-10:])
                sma_long = np.mean(closes[-20:])
                if sma_long > 0:
                    momentum = (sma_short - sma_long) / sma_long
                    signals.append(np.clip(momentum * 10, -0.5, 0.5))

            if not signals:
                return 0.0

            # Average across timeframes — if all agree, stronger signal
            avg_signal = np.mean(signals)
            agreement = 1.0 if all(s > 0 for s in signals) or all(s < 0 for s in signals) else 0.5

            return float(avg_signal * agreement)
        except Exception as e:
            logger.debug("MTF signal failed: %s", e)
            return 0.0

    def _confidence_adjustment(
        self, ohlcv: pd.DataFrame, funding: pd.DataFrame,
        ticker: Dict, news: list
    ) -> float:
        """Adjust confidence based on data quality and market conditions."""
        adj = 1.0

        # More data = higher confidence
        if len(ohlcv) < 100:
            adj *= 0.8
        if len(ohlcv) < 50:
            adj *= 0.7

        # Funding data available
        if funding.empty:
            adj *= 0.9

        # Cross-exchange data
        if not ticker:
            adj *= 0.85

        # News available
        if len(news) < 3:
            adj *= 0.95

        # High volatility = lower confidence
        if not ohlcv.empty and len(ohlcv) >= 20:
            returns = ohlcv["close"].pct_change().dropna()
            recent_vol = returns.tail(20).std()
            if recent_vol > 0.05:  # >5% daily vol
                adj *= 0.85
            elif recent_vol > 0.08:
                adj *= 0.7

        return max(0.1, min(1.0, adj))

    def retrain(self, training_data) -> None:
        """Retrain on newly logged trade outcomes."""
        if training_data is None or len(training_data) < settings.ml_min_samples_for_training:
            logger.info(
                "Not enough training samples (%s/%s) for retraining",
                len(training_data) if training_data is not None else 0,
                settings.ml_min_samples_for_training,
            )
            return

        # Convert training data to the format the ML engine expects
        ohlcv_data = training_data.get("ohlcv", pd.DataFrame())
        labels = training_data.get("labels", pd.Series())
        funding_data = training_data.get("funding", pd.DataFrame())

        if ohlcv_data.empty or labels.empty:
            return

        results = self.ml_engine.train(
            ohlcv_data=ohlcv_data,
            funding_data=funding_data if not funding_data.empty else None,
            labels=labels,
        )

        # Save retrained models
        self.ml_engine.save_all(settings.ml_model_dir)

        logger.info("Crypto signal generator retrained: %s", results)
