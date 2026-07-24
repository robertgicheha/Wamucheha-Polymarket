"""
Every category's signal generator implements this interface. The output is always
a calibrated probability, not a direction -- the risk manager and edge threshold
compare this against the market's current price, and only trade when they diverge
by more than MIN_EDGE_THRESHOLD.

Calibration matters more than raw accuracy here: a model that says "65%" and is
right 65% of the time is far more useful (and tradeable) than one that says "95%"
and is right 65% of the time.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SignalOutput:
    market_id: str
    model_probability: float  # calibrated P(YES), 0-1
    confidence: float         # model's own uncertainty estimate, 0-1
    reasoning: str            # short human-readable explanation, for the dashboard/logs


class BaseSignalGenerator(ABC):
    category: str

    @abstractmethod
    def generate(self, market_id: str) -> SignalOutput:
        """Produce a calibrated probability estimate for a given market."""
        raise NotImplementedError

    @abstractmethod
    def retrain(self, training_data) -> None:
        """Retrain on newly logged trade outcomes."""
        raise NotImplementedError
