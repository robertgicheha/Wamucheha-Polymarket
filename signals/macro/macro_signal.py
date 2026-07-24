"""
Macro economics signal generator.

Resolves on scheduled data releases / policy decisions (CPI, Fed rate, GDP, etc.)
More structured than politics but lower volume on Polymarket.

Approach:
  1. Economic calendar integration (release dates for CPI, Fed decisions, jobs reports)
  2. Historical base rates for similar decisions given similar conditions
  3. Market-implied signals from other sources (fed funds futures, bond markets)
  4. News sentiment as a secondary adjustment around the release window
"""
import logging
import math
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import settings
from data.market_parser import parse_macro_market
from data.news_feeds import NewsFeedAggregator
from data.price_feeds import DataAggregator
from signals.base_signal import BaseSignalGenerator, SignalOutput

logger = logging.getLogger(__name__)

# Historical base rates for common macro events
MACRO_BASE_RATES = {
    "fed_rate_hike": {
        "default": 0.35,
        "description": "Probability of Fed raising rates at next meeting",
    },
    "fed_rate_cut": {
        "default": 0.30,
        "description": "Probability of Fed cutting rates at next meeting",
    },
    "fed_hold": {
        "default": 0.45,
        "description": "Probability of Fed holding rates steady",
    },
    "cpi_above_expectations": {
        "default": 0.50,
        "description": "Probability CPI comes in above consensus",
    },
    "recession": {
        "default": 0.15,
        "description": "Probability of US recession within 12 months",
    },
    "gdp_positive": {
        "default": 0.75,
        "description": "Probability of positive GDP growth next quarter",
    },
    "tariff_escalation": {
        "default": 0.40,
        "description": "Probability of new tariffs being imposed",
    },
}


class MacroSignalGenerator(BaseSignalGenerator):
    category = "macro"

    def __init__(self):
        self.data = DataAggregator()
        self.news = NewsFeedAggregator()
        self.sentiment_model = None
        self.macro_base_rates = MACRO_BASE_RATES.copy()

    def generate(self, market_id: str, market_question: str = "", market_price: float = 0.5) -> SignalOutput:
        """Generate P(YES) for a macro economics market."""
        parsed = parse_macro_market(market_question)
        indicator = parsed["indicator"]
        resolution_date = parsed["resolution_date"]

        # Get macro news
        news_articles = self.news.get_market_relevant_news(
            market_question, category="macro"
        )
        headlines = [a["title"] for a in news_articles if a.get("title")]
        all_texts = [
            f"{a.get('title', '')} {a.get('description', '')}"
            for a in news_articles
        ][:30]

        # Sentiment analysis
        sentiment = self._compute_macro_sentiment(all_texts)

        # Get market-implied signals from related assets
        market_signals = self._get_market_implied_signals(indicator)

        # Base rate
        base_rate = self._get_base_rate(indicator, market_question)
        base_rate_adjusted = base_rate + sentiment["sentiment_mean"] * 0.1

        # Market-implied adjustment
        market_adj = market_signals.get("adjustment", 0.0)

        # Combine
        model_prob = base_rate_adjusted + market_adj
        model_prob = max(0.05, min(0.95, model_prob))

        # Blend with market price
        news_strength = min(len(news_articles) / 15, 1.0)
        model_weight = 0.25 + news_strength * 0.25
        market_weight = 1.0 - model_weight
        blended_prob = model_prob * model_weight + market_price * market_weight
        blended_prob = max(0.05, min(0.95, blended_prob))

        # Confidence
        base_confidence = 0.25
        data_quality = self._assess_data_quality(indicator, news_articles, market_signals)
        confidence = base_confidence * data_quality
        confidence = max(0.05, min(0.5, confidence))

        reasoning = (
            f"Indicator: {indicator} | "
            f"Base rate: {base_rate:.2f} | "
            f"Sentiment adj: {sentiment['sentiment_mean']:+.3f} | "
            f"Market signal adj: {market_adj:+.3f} | "
            f"Model: {model_prob:.3f} → blended: {blended_prob:.3f} vs market: {market_price:.3f} | "
            f"Confidence: {confidence:.2f} | "
            f"News: {len(news_articles)} articles"
        )

        return SignalOutput(
            market_id=market_id,
            model_probability=blended_prob,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _compute_macro_sentiment(self, texts: List[str]) -> Dict:
        """Compute sentiment from macro economic news."""
        if not texts:
            return {"sentiment_mean": 0.0, "n_articles": 0}

        if self.sentiment_model is None:
            try:
                from transformers import pipeline
                self.sentiment_model = pipeline(
                    "sentiment-analysis",
                    model=settings.ml_sentiment_model,
                    top_k=None,
                    truncation=True,
                    max_length=512,
                )
            except Exception:
                return {"sentiment_mean": 0.0, "n_articles": len(texts)}

        sentiments = []
        for text in texts[:15]:
            try:
                result = self.sentiment_model(text[:512])
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        result = result[0]
                    scores = {r["label"].lower(): r["score"] for r in result}
                    compound = scores.get("positive", 0) - scores.get("negative", 0)
                    sentiments.append(compound)
            except Exception:
                continue

        return {
            "sentiment_mean": float(np.mean(sentiments)) if sentiments else 0.0,
            "n_articles": len(texts),
        }

    def _get_market_implied_signals(self, indicator: str) -> Dict:
        """
        Get market-implied signals from related financial instruments.
        For Fed rate: fed funds futures, 2Y/10Y treasury yield curve
        For CPI: breakeven inflation rates
        For recession: yield curve inversion
        """
        signals = {}
        adjustment = 0.0

        if indicator in ("fed_rate",):
            # 2-year treasury yield as proxy for rate expectations
            # Higher yield → market expects rates to stay/go higher
            try:
                binance_data = self.data.binance
                # Use BTC as a proxy for risk sentiment (not ideal but available)
                ticker = binance_data.get_ticker("BTCUSDT")
                if ticker and ticker.get("change_24h_pct") is not None:
                    # Risk-on sentiment might correlate with dovish Fed expectations
                    change_pct = ticker["change_24h_pct"]
                    if change_pct > 3:
                        adjustment = 0.02  # risk-on → slightly dovish
                    elif change_pct < -3:
                        adjustment = -0.02  # risk-off → slightly hawkish
            except Exception:
                pass

        elif indicator in ("recession",):
            try:
                # Check yield curve proxy via BTC volatility as risk indicator
                binance_data = self.data.binance
                ohlcv = binance_data.get_ohlcv("BTCUSDT", "1d", 30)
                if ohlcv and len(ohlcv) >= 10:
                    closes = [c["close"] for c in ohlcv]
                    returns = np.diff(closes) / closes[:-1]
                    vol = np.std(returns[-20:])
                    if vol > 0.04:
                        adjustment = 0.03  # high vol → slightly more recession risk
                    elif vol < 0.015:
                        adjustment = -0.02  # low vol → less recession risk
            except Exception:
                pass

        signals["adjustment"] = adjustment
        return signals

    def _get_base_rate(self, indicator: str, question: str) -> float:
        """Get a base rate prior for the macro event."""
        # Try to match indicator to known base rates
        q_lower = question.lower()
        for key, rate_info in self.macro_base_rates.items():
            key_words = key.replace("_", " ")
            if any(w in q_lower for w in key_words.split()):
                return rate_info["default"]

        return 0.5  # uninformative prior

    def _assess_data_quality(
        self, indicator: str, articles: List, signals: Dict
    ) -> float:
        """Assess the quality and quantity of available data."""
        score = 1.0

        # More news = better data
        if len(articles) > 10:
            score *= 1.0
        elif len(articles) > 5:
            score *= 0.9
        elif len(articles) > 0:
            score *= 0.8
        else:
            score *= 0.7

        # Market signals available
        if signals.get("adjustment") != 0:
            score *= 1.0
        else:
            score *= 0.9

        return max(0.3, min(1.0, score))

    def retrain(self, training_data) -> None:
        """Update base rates from resolved macro markets."""
        if training_data is None:
            return
        resolved = training_data.get("resolved_markets", [])
        for market in resolved:
            indicator = market.get("indicator", "")
            outcome = market.get("outcome", None)
            if indicator and outcome is not None:
                # Update base rate with exponential moving average
                key = f"{indicator}"
                if key in self.macro_base_rates:
                    current = self.macro_base_rates[key]["default"]
                    new_rate = 1.0 if outcome else 0.0
                    alpha = 0.1
                    self.macro_base_rates[key]["default"] = (
                        current * (1 - alpha) + new_rate * alpha
                    )
        logger.info("Macro base rates updated")
