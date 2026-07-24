"""
Politics/current-events signal generator.

Approach:
  1. News/event stream (NewsAPI, GDELT, RSS) filtered to the market's topic
  2. Transformer-based sentiment + entity extraction on relevant articles
  3. Base-rate modeling: how similar past events resolved, adjusted for current signal
  4. Where available, aggregate other forecasting sources (polling averages, other
     prediction markets) as a prior
  5. Confidence is LOW by default — these markets have much noisier ground truth

These markets resolve on subjective real-world outcomes with no clean price feed
as ground truth, so the signal is fundamentally different from crypto.
"""
import logging
import math
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from config.settings import settings
from data.market_parser import parse_election_market
from data.news_feeds import NewsFeedAggregator
from ml.engine import EnsembleEngine, FeatureEngine
from signals.base_signal import BaseSignalGenerator, SignalOutput

logger = logging.getLogger(__name__)

# Base rate priors for common political events
# These represent historical frequencies and should be updated as new data arrives
BASE_RATES = {
    "presidential_incumbent_win": 0.65,  # incumbents historically win ~65%
    "senate_majority_change": 0.25,       # average chance of flip in a cycle
    "governor_incumbent_win": 0.70,
    "legislation_pass": 0.40,             # bills introduced vs enacted
    "cabinet_confirmation": 0.85,
    "impeachment_conviction": 0.10,
}


class PoliticsSignalGenerator(BaseSignalGenerator):
    category = "politics"

    def __init__(self):
        self.news = NewsFeedAggregator()
        self.feature_engine = FeatureEngine()
        self.sentiment_model = None
        self.base_rate_priors = BASE_RATES.copy()

    def generate(self, market_id: str, market_question: str = "", market_price: float = 0.5) -> SignalOutput:
        """Generate P(YES) for a politics/current-events market."""
        parsed = parse_election_market(market_question)
        event_type = parsed["election_type"]
        candidates = parsed["candidates"]
        resolution_date = parsed["resolution_date"]

        # Fetch relevant news
        news_articles = self.news.get_market_relevant_news(
            market_question, category="politics"
        )
        headlines = [a["title"] for a in news_articles if a.get("title")]
        all_texts = [
            f"{a.get('title', '')} {a.get('description', '')}"
            for a in news_articles
        ][:30]

        # Sentiment analysis
        sentiment_features = self._compute_sentiment(all_texts)

        # Entity/faction analysis
        entity_signal = self._compute_entity_signal(
            news_articles, candidates
        )

        # Base rate prior
        base_rate = self._get_base_rate(event_type, parsed)

        # Market-implied prior from other sources
        market_prior = market_price

        # Combine signals using Bayesian-inspired framework
        # Start with base rate, adjust by news sentiment, then blend with market
        sentiment_adjustment = sentiment_features["sentiment_mean"] * 0.15  # max 15% adjustment
        entity_adjustment = entity_signal * 0.10  # max 10% adjustment

        # Model probability = adjusted base rate
        model_prob = base_rate + sentiment_adjustment + entity_adjustment
        model_prob = max(0.05, min(0.95, model_prob))

        # Blend with market prior (don't try to completely out-predict the market)
        # Weight model more when we have strong news signal
        news_strength = min(len(news_articles) / 20, 1.0)
        model_weight = 0.3 + news_strength * 0.3  # 30-60% model, 40-70% market
        market_weight = 1.0 - model_weight
        blended_prob = model_prob * model_weight + market_prior * market_weight
        blended_prob = max(0.05, min(0.95, blended_prob))

        # Confidence: politics markets are inherently noisier
        base_confidence = 0.2  # low base for politics
        news_bonus = min(len(news_articles) / 30, 0.2)
        recency_bonus = self._recency_bonus(news_articles)
        sentiment_consistency = self._sentiment_consistency(sentiment_features)

        confidence = base_confidence + news_bonus + recency_bonus + sentiment_consistency
        confidence = max(0.05, min(0.6, confidence))  # cap at 0.6 for politics

        reasoning = (
            f"Event: {event_type}, Candidates: {candidates} | "
            f"Base rate: {base_rate:.2f} | "
            f"Sentiment adj: {sentiment_adjustment:+.3f} | "
            f"Entity signal: {entity_adjustment:+.3f} | "
            f"Model: {model_prob:.3f} → blended: {blended_prob:.3f} vs market: {market_price:.3f} | "
            f"Confidence: {confidence:.2f} | "
            f"News articles: {len(news_articles)}"
        )

        return SignalOutput(
            market_id=market_id,
            model_probability=blended_prob,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _compute_sentiment(self, texts: List[str]) -> Dict:
        """Compute sentiment features from news texts."""
        if not texts:
            return {
                "sentiment_mean": 0.0,
                "sentiment_std": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "n_articles": 0,
            }

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
            except Exception as e:
                logger.warning("Failed to load sentiment model: %s", e)
                return {
                    "sentiment_mean": 0.0, "sentiment_std": 0.0,
                    "positive_ratio": 0.0, "negative_ratio": 0.0,
                    "n_articles": len(texts),
                }

        sentiments = []
        for text in texts[:20]:
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

        if not sentiments:
            return {
                "sentiment_mean": 0.0, "sentiment_std": 0.0,
                "positive_ratio": 0.0, "negative_ratio": 0.0,
                "n_articles": len(texts),
            }

        arr = np.array(sentiments)
        return {
            "sentiment_mean": float(np.mean(arr)),
            "sentiment_std": float(np.std(arr)) if len(arr) > 1 else 0.0,
            "positive_ratio": float(np.mean(arr > 0.1)),
            "negative_ratio": float(np.mean(arr < -0.1)),
            "n_articles": len(texts),
        }

    def _compute_entity_signal(self, articles: List[Dict], candidates: List[str]) -> float:
        """
        Compute signal from entity/faction mentions in news.
        More positive mentions for a candidate → slight boost for their side.
        """
        if not articles or not candidates:
            return 0.0

        signals = []
        for candidate in candidates[:3]:
            mentions = 0
            positive = 0
            negative = 0
            for a in articles:
                text = f"{a.get('title', '')} {a.get('description', '')}".lower()
                if candidate.lower() in text:
                    mentions += 1
                    tone = a.get("tone", 0)
                    if isinstance(tone, (int, float)):
                        if tone > 2:
                            positive += 1
                        elif tone < -2:
                            negative += 1

            if mentions > 0:
                candidate_signal = (positive - negative) / mentions
                signals.append(candidate_signal)

        return float(np.mean(signals)) if signals else 0.0

    def _get_base_rate(self, event_type: str, parsed: Dict) -> float:
        """Get a base rate prior for the event type."""
        if event_type == "presidential":
            return self.base_rate_priors.get("presidential_incumbent_win", 0.5)
        elif event_type == "senate":
            return self.base_rate_priors.get("senate_majority_change", 0.5)
        elif event_type == "governor":
            return self.base_rate_priors.get("governor_incumbent_win", 0.5)
        return 0.5  # uninformative prior

    def _recency_bonus(self, articles: List[Dict]) -> float:
        """More recent articles → slightly higher confidence."""
        if not articles:
            return 0.0
        now = datetime.utcnow()
        recent = 0
        for a in articles[:10]:
            pub_str = a.get("published_at", "")
            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                hours_ago = (now - pub_dt.replace(tzinfo=None)).total_seconds() / 3600
                if hours_ago < 24:
                    recent += 1
            except (ValueError, TypeError):
                continue
        return min(recent / 10, 0.15)

    def _sentiment_consistency(self, features: Dict) -> float:
        """If sentiment is very one-sided, boost confidence slightly."""
        if features["n_articles"] < 5:
            return 0.0
        dominant = max(features["positive_ratio"], features["negative_ratio"])
        return min((dominant - 0.5) * 0.3, 0.15) if dominant > 0.5 else 0.0

    def retrain(self, training_data) -> None:
        """Update base rates based on resolved markets."""
        if training_data is None:
            return
        # Base rates can be updated from resolved market data
        # This is a placeholder for the actual implementation
        logger.info("Politics base rate retraining not yet implemented")
