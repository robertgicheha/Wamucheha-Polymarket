"""
Bridges BRTIEngine's tick history into the DataFrame schema
ml/btc_features.py::BTCFeatureEngine expects, so the live trading loop and
the TTE retrain loop can compute real features instead of the placeholder
random noise / hardcoded values that used to feed the model.

BRTI ticks are orderbook-derived (no per-trade tape), so trade-level columns
("size", "side", "trade_count") are intentionally omitted — BTCFeatureEngine
already degrades gracefully when a column is missing (checks `if col in
df.columns`, fills NaN with 0). Feature groups that need a trade tape (VPIN,
Kyle's Lambda, buy/sell imbalance) will read as zero until a live trade-tape
recorder is added.
"""
import logging
from typing import List, Optional

import pandas as pd

from data.brti.brti_engine import BRTIEngine, BRTITick
from ml.btc_features import BTCFeatureEngine

logger = logging.getLogger(__name__)

_feature_engine = BTCFeatureEngine()


def ticks_to_frame(ticks: List[BRTITick]) -> pd.DataFrame:
    """Map a list of BRTITick into the column schema BTCFeatureEngine expects."""
    if not ticks:
        return pd.DataFrame()

    rows = []
    for t in ticks:
        rows.append({
            "timestamp": t.timestamp,
            "price": t.brti_price,
            "mid_price": t.brti_price,
            "bid1_price": t.mid_price_bid,
            "ask1_price": t.mid_price_ask,
            "bid_depth_5": t.bid_volume,
            "ask_depth_5": t.ask_volume,
            "bid_depth_10": t.bid_volume,
            "ask_depth_10": t.ask_volume,
            "spread": t.brti_price * t.spread_bps / 10000.0 if t.brti_price else 0.0,
        })
    return pd.DataFrame(rows)


def build_feature_frame(
    brti_engine: BRTIEngine, lookback: int = 1000
) -> Optional[pd.DataFrame]:
    """
    Compute the full ~80-feature DataFrame from BRTIEngine's recent tick
    history. Returns None if there isn't enough history yet (feature
    engine needs a minimum window for rolling stats to be meaningful).
    """
    history = brti_engine.tick_history[-lookback:]
    if len(history) < 60:
        return None

    raw = ticks_to_frame(history)
    features = _feature_engine.compute_all_features(raw)
    if features.empty:
        return None
    return features


def raw_price_frame(brti_engine: BRTIEngine, lookback: Optional[int] = None) -> pd.DataFrame:
    """
    Raw (pre-feature) price/orderbook DataFrame for training —
    ml/tte_orchestrator.py computes features itself from this shape.
    """
    history = brti_engine.tick_history
    if lookback:
        history = history[-lookback:]
    return ticks_to_frame(history)
