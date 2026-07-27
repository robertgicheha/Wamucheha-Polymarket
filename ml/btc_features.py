"""
BTC feature engineering — ~80 features from orderbook, trade, and market data.

Feature categories:
  1. Trade data features (20) — volume, aggression, trade size distribution
  2. Orderbook features (15) — depth, imbalance, levels
  3. Cross-exchange features (8) — correlation, divergence, lead-lag
  4. Microstructure features (12) — VPIN, Kyle's Lambda, Amihud, Roll spread
  5. Momentum / technical features (15) — RSI, MACD, Bollinger, ATR
  6. Time-based features (10) — hour, day, session, proximity to events

NO Kalshi features in Layer 1 — only BTC spot/microstructure data.
Kalshi features are added in Layer 2.
"""
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPSILON = 1e-10


class BTCFeatureEngine:
    """
    Computes ~80 BTC features from raw market data for ML prediction.

    Input: DataFrame with columns from orderbook/trade feed:
        timestamp, price, size, side (buy/sell),
        bid1_price, bid1_size, ask1_price, ask1_size,
        bid_depth_5, ask_depth_5, bid_depth_10, ask_depth_10,
        spread, mid_price, last_trade_side, trade_count,
        exchange (optional, for cross-exchange features)

    Output: DataFrame with ~80 feature columns.
    """

    def __init__(self, lookback_windows: Optional[List[int]] = None):
        self.lookback_windows = lookback_windows or [5, 10, 20, 50, 100]

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all feature categories and return a single DataFrame."""
        if df.empty or len(df) < 10:
            logger.warning("Insufficient data for feature computation (%d rows)", len(df))
            return pd.DataFrame()

        df = df.copy()
        features = pd.DataFrame(index=df.index)

        # 1. Trade data features
        trade_features = self._trade_features(df)
        features = pd.concat([features, trade_features], axis=1)

        # 2. Orderbook features
        ob_features = self._orderbook_features(df)
        features = pd.concat([features, ob_features], axis=1)

        # 3. Cross-exchange features (if multi-exchange data available)
        if "exchange" in df.columns and df["exchange"].nunique() > 1:
            cross_features = self._cross_exchange_features(df)
            features = pd.concat([features, cross_features], axis=1)

        # 4. Microstructure features
        micro_features = self._microstructure_features(df)
        features = pd.concat([features, micro_features], axis=1)

        # 5. Momentum / technical features
        momentum_features = self._momentum_features(df)
        features = pd.concat([features, momentum_features], axis=1)

        # 6. Time-based features
        time_features = self._time_features(df)
        features = pd.concat([features, time_features], axis=1)

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        logger.info(
            "Computed %d features from %d rows",
            len(features.columns), len(features),
        )
        return features

    # ── 1. Trade Data Features (20) ───────────────────────────────────

    def _trade_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features derived from individual trade data:
        - Volume, trade count, buy/sell ratio
        - Average trade size, VWAP
        - Trade aggression indicators
        """
        f = pd.DataFrame(index=df.index)

        # Basic price/return features
        if "price" in df.columns:
            f["return_1"] = df["price"].pct_change(1)
            f["log_return_1"] = np.log(df["price"] / df["price"].shift(1).replace(0, EPSILON))

            for w in self.lookback_windows:
                f[f"return_{w}"] = df["price"].pct_change(w)
                f[f"log_return_{w}"] = np.log(
                    df["price"] / df["price"].shift(w).replace(0, EPSILON)
                )

        # Volume features
        if "size" in df.columns:
            f["volume_1"] = df["size"]
            for w in self.lookback_windows:
                f[f"volume_sum_{w}"] = df["size"].rolling(w).sum()
                f[f"volume_mean_{w}"] = df["size"].rolling(w).mean()
                f[f"volume_std_{w}"] = df["size"].rolling(w).std()

            # Volume-weighted average price
            if "price" in df.columns:
                vwap_num = (df["price"] * df["size"]).rolling(20).sum()
                vwap_den = df["size"].rolling(20).sum().replace(0, EPSILON)
                f["vwap_20"] = vwap_num / vwap_den
                f["price_vs_vwap"] = (df["price"] - f["vwap_20"]) / f["vwap_20"].replace(0, EPSILON)

        # Buy/sell ratio
        if "side" in df.columns:
            buy_mask = df["side"].str.lower().isin(["buy", "bid", "b"])
            sell_mask = df["side"].str.lower().isin(["sell", "ask", "s"])

            f["buy_volume"] = df["size"].where(buy_mask, 0).rolling(20).sum()
            f["sell_volume"] = df["size"].where(sell_mask, 0).rolling(20).sum()
            total = (f["buy_volume"] + f["sell_volume"]).replace(0, EPSILON)
            f["buy_sell_ratio"] = f["buy_volume"] / f["sell_volume"].replace(0, EPSILON)
            f["buy_sell_imbalance"] = (f["buy_volume"] - f["sell_volume"]) / total

            # Aggressive buying/selling
            f["aggressive_buy_pct"] = df["size"].where(buy_mask, 0).rolling(50).sum() / total
            f["aggressive_sell_pct"] = df["size"].where(sell_mask, 0).rolling(50).sum() / total

        # Trade count
        if "trade_count" in df.columns:
            f["trade_count_1"] = df["trade_count"]
            for w in [10, 20, 50]:
                f[f"trade_count_sum_{w}"] = df["trade_count"].rolling(w).sum()
                f[f"avg_trade_size_{w}"] = (
                    df["size"].rolling(w).sum() /
                    df["trade_count"].rolling(w).sum().replace(0, EPSILON)
                )

        return f

    # ── 2. Orderbook Features (15) ────────────────────────────────────

    def _orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features from orderbook depth and structure.
        """
        f = pd.DataFrame(index=df.index)

        # Spread
        if "spread" in df.columns:
            f["spread"] = df["spread"]
            f["spread_bps"] = df["spread"] / df["mid_price"].replace(0, EPSILON) * 10000
            for w in [10, 20, 50]:
                f[f"spread_mean_{w}"] = df["spread"].rolling(w).mean()
                f[f"spread_std_{w}"] = df["spread"].rolling(w).std()

        # Depth imbalance
        if "bid_depth_5" in df.columns and "ask_depth_5" in df.columns:
            total_5 = df["bid_depth_5"] + df["ask_depth_5"]
            f["depth_imbalance_5"] = (
                (df["bid_depth_5"] - df["ask_depth_5"]) / total_5.replace(0, EPSILON)
            )
            f["depth_ratio_5"] = df["bid_depth_5"] / df["ask_depth_5"].replace(0, EPSILON)

        if "bid_depth_10" in df.columns and "ask_depth_10" in df.columns:
            total_10 = df["bid_depth_10"] + df["ask_depth_10"]
            f["depth_imbalance_10"] = (
                (df["bid_depth_10"] - df["ask_depth_10"]) / total_10.replace(0, EPSILON)
            )
            f["depth_ratio_10"] = df["bid_depth_10"] / df["ask_depth_10"].replace(0, EPSILON)

        # Best bid/ask sizes
        if "bid1_size" in df.columns and "ask1_size" in df.columns:
            total_best = df["bid1_size"] + df["ask1_size"]
            f["best_bid_ask_ratio"] = df["bid1_size"] / df["ask1_size"].replace(0, EPSILON)
            f["best_imbalance"] = (
                (df["bid1_size"] - df["ask1_size"]) / total_best.replace(0, EPSILON)
            )

        # Mid price features
        if "mid_price" in df.columns:
            f["mid_return_1"] = df["mid_price"].pct_change(1)
            for w in [5, 10, 20]:
                f[f"mid_volatility_{w}"] = f["mid_return_1"].rolling(w).std()

        return f

    # ── 3. Cross-Exchange Features (8) ────────────────────────────────

    def _cross_exchange_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features comparing price/orderbook across exchanges.
        Only computed when multi-exchange data is available.
        """
        f = pd.DataFrame(index=df.index)

        if "exchange" not in df.columns or "price" not in df.columns:
            return f

        exchanges = df["exchange"].unique()
        if len(exchanges) < 2:
            return f

        # Cross-exchange price spread
        pivot = df.pivot_table(values="price", index=df.index, columns="exchange")
        if len(pivot.columns) >= 2:
            ex_a, ex_b = pivot.columns[0], pivot.columns[1]
            price_diff = pivot[ex_a] - pivot[ex_b]
            f["cross_spread_abs"] = price_diff.abs()
            f["cross_spread_bps"] = (
                price_diff.abs() / pivot.mean(axis=1).replace(0, EPSILON) * 10000
            )
            f["cross_spread_sign"] = np.sign(price_diff)

            # Lead-lag: which exchange leads price changes?
            f["cross_lead_lag"] = (
                pivot[ex_a].pct_change(1) - pivot[ex_b].pct_change(1)
            )

        # Cross-exchange correlation
        if len(pivot.columns) >= 2:
            returns_a = pivot[ex_a].pct_change()
            returns_b = pivot[ex_b].pct_change()
            f["cross_correlation"] = returns_a.rolling(50).corr(returns_b)

        # Cross-exchange divergence
        if "mid_price" in df.columns:
            per_exchange_mean = df.groupby("exchange")["price"].transform("mean")
            f["price_vs_exchange_mean"] = df["price"] - per_exchange_mean

        return f

    # ── 4. Microstructure Features (12) ───────────────────────────────

    def _microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Market microstructure features:
        - VPIN (Volume-Synchronized Probability of Informed Trading)
        - Kyle's Lambda (price impact)
        - Amihud illiquidity
        - Roll spread estimator
        - Realized spread
        """
        f = pd.DataFrame(index=df.index)

        if "price" not in df.columns or "size" not in df.columns:
            return f

        returns = df["price"].pct_change()

        # VPIN (simplified)
        if "side" in df.columns:
            buy_mask = df["side"].str.lower().isin(["buy", "bid", "b"])
            sell_mask = df["side"].str.lower().isin(["sell", "ask", "s"])

            buy_vol = df["size"].where(buy_mask, 0)
            sell_vol = df["size"].where(sell_mask, 0)

            for w in [20, 50]:
                buy_sum = buy_vol.rolling(w).sum()
                sell_sum = sell_vol.rolling(w).sum()
                total = (buy_sum + sell_sum).replace(0, EPSILON)
                f[f"vpin_{w}"] = (buy_sum - sell_sum).abs() / total

        # Kyle's Lambda (price impact): Cov(dP, V) / Var(V)
        if "side" in df.columns:
            signed_volume = df["size"].where(
                df["side"].str.lower().isin(["buy", "bid", "b"]), -df["size"]
            )
            for w in [20, 50]:
                cov = returns.rolling(w).cov(signed_volume)
                var = signed_volume.rolling(w).var().replace(0, EPSILON)
                f[f"kyle_lambda_{w}"] = cov / var

        # Amihud illiquidity: |return| / dollar volume
        if "mid_price" in df.columns:
            dollar_volume = df["size"] * df["mid_price"]
            for w in [20, 50]:
                f[f"amihud_{w}"] = (
                    returns.abs().rolling(w).mean() /
                    dollar_volume.rolling(w).mean().replace(0, EPSILON)
                )

        # Roll spread estimator: 2 * sqrt(-Cov(r_t, r_{t-1}))
        for w in [20, 50]:
            lagged_returns = returns.shift(1)
            cov = returns.rolling(w).cov(lagged_returns)
            # Roll spread is only defined when covariance is negative
            roll = np.where(cov < 0, 2 * np.sqrt(-cov), 0)
            f[f"roll_spread_{w}"] = roll

        # Realized spread (simplified): spread - 2 * price impact
        if "spread" in df.columns and "kyle_lambda_20" in f.columns:
            f["realized_spread"] = df["spread"] - 2 * f["kyle_lambda_20"].abs()

        return f

    # ── 5. Momentum / Technical Features (15) ─────────────────────────

    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Technical indicators and momentum features.
        """
        f = pd.DataFrame(index=df.index)

        if "price" not in df.columns:
            return f

        close = df["price"]

        # RSI
        for period in [14, 28]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss.replace(0, EPSILON)
            f[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        f["macd"] = ema12 - ema26
        f["macd_signal"] = f["macd"].ewm(span=9).mean()
        f["macd_hist"] = f["macd"] - f["macd_signal"]

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        f["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, EPSILON)
        f["bb_width"] = (bb_upper - bb_lower) / sma20.replace(0, EPSILON)

        # ATR (Average True Range)
        high = df.get("high", close)
        low = df.get("low", close)
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        f["atr_14"] = tr.rolling(14).mean()

        # Stochastic Oscillator
        for period in [14]:
            low_min = low.rolling(period).min()
            high_max = high.rolling(period).max()
            f[f"stoch_k_{period}"] = (
                (close - low_min) / (high_max - low_min).replace(0, EPSILON) * 100
            )
            f[f"stoch_d_{period}"] = f[f"stoch_k_{period}"].rolling(3).mean()

        # Rate of change
        for period in [5, 10, 20]:
            f[f"roc_{period}"] = close.pct_change(period)

        # Momentum
        for period in [10, 20]:
            f[f"momentum_{period}"] = close / close.shift(period) - 1

        return f

    # ── 6. Time-Based Features (10) ───────────────────────────────────

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cyclical time features for intraday patterns.
        Uses sin/cos encoding for hour, minute, day-of-week.
        """
        f = pd.DataFrame(index=df.index)

        if "timestamp" not in df.columns:
            return f

        ts = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        if ts.isna().all():
            return f

        # Hour of day (0-23)
        hour = ts.dt.hour
        f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # Minute of hour (0-59)
        minute = ts.dt.minute
        f["minute_sin"] = np.sin(2 * np.pi * minute / 60)
        f["minute_cos"] = np.cos(2 * np.pi * minute / 60)

        # Day of week (0=Monday, 6=Sunday)
        dow = ts.dt.dayofweek
        f["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        f["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        # Session indicators (UTC-based)
        f["is_asia_session"] = ((hour >= 0) & (hour < 8)).astype(float)
        f["is_europe_session"] = ((hour >= 7) & (hour < 16)).astype(float)
        f["is_us_session"] = ((hour >= 13) & (hour < 22)).astype(float)

        return f

    # ── Feature Names ──────────────────────────────────────────────────

    def get_feature_names(self) -> List[str]:
        """Return list of all feature names this engine produces."""
        dummy = pd.DataFrame({
            "timestamp": [0] * 100,
            "price": np.random.randn(100) * 100 + 100000,
            "size": np.abs(np.random.randn(100)),
            "side": ["buy"] * 50 + ["sell"] * 50,
            "bid1_price": [100000] * 100,
            "bid1_size": [1.0] * 100,
            "ask1_price": [100001] * 100,
            "ask1_size": [1.0] * 100,
            "bid_depth_5": [5.0] * 100,
            "ask_depth_5": [5.0] * 100,
            "bid_depth_10": [10.0] * 100,
            "ask_depth_10": [10.0] * 100,
            "spread": [1.0] * 100,
            "mid_price": [100000.5] * 100,
            "trade_count": [10] * 100,
        })
        features = self.compute_all_features(dummy)
        return list(features.columns)
