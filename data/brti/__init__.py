"""
BRTI (Bitcoin Real-Time Index) replication engine.

Reproduces the CF Benchmarks BRTI methodology:
  1. Subscribes to orderbook WebSockets on constituent exchanges
  2. Aggregates orderbooks into a consolidated orderbook
  3. Applies order size cap (100 BTC)
  4. Builds mid price-volume curves
  5. Calculates utilized depth (max volume where deviation <= 0.5%)
  6. Applies exponential weighting (lambda = 1/5000)
  7. Computes BRTI as weighted sum of mid price-volume curve
  8. Publishes at 1-second frequency

Kalshi settlement: 60-second average of BRTI ticks at contract close.
"""
from data.brti.brti_engine import BRTIEngine
from data.brti.kalshi_settlement import KalshiSettlementCalculator

__all__ = ["BRTIEngine", "KalshiSettlementCalculator"]
