"""
Sports signal generator.

Different pipeline from crypto/politics — needs team/player stats feeds,
not news sentiment as the primary input.

Approach:
  1. Team/player stats feed (team records, recent performance, head-to-head)
  2. Elo-style rating model, updated per result
  3. Injury/roster news as an adjustment layer (lighter NLP)
  4. Compare model probability against both Polymarket price AND other sportsbook
     lines where available
"""
import logging
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import settings
from data.market_parser import parse_sports_market
from data.news_feeds import NewsFeedAggregator
from signals.base_signal import BaseSignalGenerator, SignalOutput

logger = logging.getLogger(__name__)


class EloModel:
    """Simple Elo rating system for team ranking."""

    def __init__(self, k_factor: float = 20, default_elo: float = 1500):
        self.k_factor = k_factor
        self.default_elo = default_elo
        self.ratings: Dict[str, float] = {}
        self.history: List[Dict] = []

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.default_elo)

    def expected_score(self, team_a: str, team_b: str) -> float:
        """Expected probability of team_a winning."""
        elo_a = self.get_rating(team_a)
        elo_b = self.get_rating(team_b)
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))

    def update(self, winner: str, loser: str, draw: bool = False) -> None:
        """Update ratings after a match."""
        if draw:
            score_a, score_b = 0.5, 0.5
        else:
            score_a, score_b = 1.0, 0.0

        expected = self.expected_score(winner, loser)
        self.ratings[winner] = self.get_rating(winner) + self.k_factor * (score_a - expected)
        self.ratings[loser] = self.get_rating(loser) + self.k_factor * (score_b - (1 - expected))

    def predict_matchup(self, team_a: str, team_b: str) -> Dict:
        """Predict the outcome of a matchup."""
        prob_a = self.expected_score(team_a, team_b)
        return {
            "team_a": team_a,
            "team_b": team_b,
            "prob_a_wins": prob_a,
            "prob_b_wins": 1 - prob_a,
            "elo_a": self.get_rating(team_a),
            "elo_b": self.get_rating(team_b),
        }


class SportsSignalGenerator(BaseSignalGenerator):
    category = "sports"

    def __init__(self):
        self.elo = EloModel()
        self.news = NewsFeedAggregator()
        self.sentiment_model = None

    def generate(self, market_id: str, market_question: str = "", market_price: float = 0.5) -> SignalOutput:
        """Generate P(YES) for a sports market."""
        parsed = parse_sports_market(market_question)
        sport = parsed["sport"]
        teams = parsed["teams"]

        if not teams or len(teams) < 2:
            return SignalOutput(
                market_id=market_id,
                model_probability=market_price,
                confidence=0.1,
                reasoning=f"Cannot parse teams from: {market_question}",
            )

        team_a, team_b = teams[0], teams[1]

        # Get Elo prediction
        elo_prediction = self.elo.predict_matchup(team_a, team_b)
        model_prob = elo_prediction["prob_a_wins"]

        # Check if market asks for team_a winning (YES) or team_b (NO)
        # Simple heuristic: if team_a is mentioned first or is the "home" team
        # the YES outcome is likely team_a winning
        q_lower = market_question.lower()
        if team_b.lower() in q_lower:
            # Check if the question asks about team_b specifically
            if f"will {team_b.lower()}" in q_lower or f"{team_b.lower()} win" in q_lower:
                model_prob = elo_prediction["prob_b_wins"]

        # News/injury adjustment
        injury_adjustment = self._compute_injury_adjustment(
            teams, market_question
        )
        model_prob += injury_adjustment
        model_prob = max(0.05, min(0.95, model_prob))

        # Recent form adjustment (if we have historical data)
        form_adjustment = self._compute_form_adjustment(team_a, team_b)
        model_prob += form_adjustment
        model_prob = max(0.05, min(0.95, model_prob))

        # Confidence
        elo_gap = abs(elo_prediction["elo_a"] - elo_prediction["elo_b"])
        elo_confidence = min(elo_gap / 400, 0.3)  # bigger gap = more confident
        base_confidence = 0.3  # moderate base for sports
        news_bonus = 0.1 if injury_adjustment != 0 else 0

        confidence = base_confidence + elo_confidence + news_bonus
        confidence = max(0.1, min(0.7, confidence))

        reasoning = (
            f"Sport: {sport}, Teams: {team_a} vs {team_b} | "
            f"Elo: {elo_prediction['elo_a']:.0f} vs {elo_prediction['elo_b']:.0f} | "
            f"Elo prob: {elo_prediction['prob_a_wins']:.3f} | "
            f"Injury adj: {injury_adjustment:+.3f} | "
            f"Form adj: {form_adjustment:+.3f} | "
            f"Final: {model_prob:.3f} vs market: {market_price:.3f} | "
            f"Confidence: {confidence:.2f}"
        )

        return SignalOutput(
            market_id=market_id,
            model_probability=model_prob,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _compute_injury_adjustment(
        self, teams: List[str], question: str
    ) -> float:
        """Check news for injury reports that could affect the outcome."""
        news = self.news.get_market_relevant_news(question, category="sports")
        injury_keywords = [
            "injury", "injured", "out", "ruled out", "doubtful",
            "questionable", "suspended", "suspension", "sidelined",
        ]

        adjustment = 0.0
        for article in news[:10]:
            text = f"{article.get('title', '')} {article.get('description', '')}".lower()
            for team in teams:
                if team.lower() in text:
                    for kw in injury_keywords:
                        if kw in text:
                            # Injury to a key player = slight boost for the other team
                            if teams.index(team) == 0:
                                adjustment -= 0.03
                            else:
                                adjustment += 0.03
                            break
        return adjustment

    def _compute_form_adjustment(self, team_a: str, team_b: str) -> float:
        """
        Recent form adjustment based on win/loss streaks.
        This is a placeholder — real implementation would need a stats API.
        """
        # In a real implementation, this would pull from a sports data API
        # For now, return 0 (no adjustment)
        return 0.0

    def update_elo(self, winner: str, loser: str, draw: bool = False) -> None:
        """Update Elo ratings after a known result. Called from backtest or live feed."""
        self.elo.update(winner, loser, draw)

    def load_ratings(self, ratings: Dict[str, float]) -> None:
        """Load pre-computed Elo ratings."""
        self.elo.ratings = ratings

    def retrain(self, training_data) -> None:
        """Update Elo ratings from resolved matches."""
        if training_data is None:
            return
        matches = training_data.get("matches", [])
        for match in matches:
            if match.get("winner") and match.get("loser"):
                draw = match.get("draw", False)
                self.elo.update(match["winner"], match["loser"], draw)
        logger.info("Sports Elo model updated with %d matches", len(matches))
