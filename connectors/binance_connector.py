"""
Binance connector for OHLCV candles and perpetual futures funding rates.
No authentication needed for market data — Binance public API is free and
well-documented. Used alongside OKX for cross-exchange signal validation.

Endpoints:
  - Klines (OHLCV): GET https://api.binance.com/api/v3/klines
  - Funding rate history: GET https://fapi.binance.com/fapi/v1/fundingRate
  - Current funding rate: GET https://fapi.binance.com/fapi/v1/premiumIndex
"""
import logging
import time
from typing import Dict, List, Optional

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

BINANCE_SPOT_BASE = "https://api.binance.com"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"


class BinanceConnector:
    def __init__(self):
        self.api_key = settings.binance_api_key
        self.api_secret = settings.binance_api_secret
        self._session = requests.Session()
        if self.api_key:
            self._session.headers["X-MBX-APIKEY"] = self.api_key

    # ── Spot OHLCV ────────────────────────────────────────────────────

    def get_ohlcv(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict]:
        """
        Fetch OHLCV candles from Binance spot.
        symbol: "BTCUSDT", "ETHUSDT", "SOLUSDT"
        interval: "1m","5m","15m","1h","4h","1d","1w"
        Returns list of dicts with keys: ts, open, high, low, close, volume, close_time
        """
        try:
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time
            resp = self._session.get(
                f"{BINANCE_SPOT_BASE}/api/v3/klines", params=params, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            candles = []
            for c in data:
                candles.append({
                    "ts": int(c[0]),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                    "close_time": int(c[6]),
                    "quote_volume": float(c[7]),
                    "trades": int(c[8]),
                    "taker_buy_base": float(c[9]),
                    "taker_buy_quote": float(c[10]),
                })
            return candles
        except requests.RequestException as e:
            logger.error("Binance OHLCV request failed: %s", e)
            return []

    def get_ohlcv_multi_timeframe(
        self, symbol: str = "BTCUSDT"
    ) -> Dict[str, List[Dict]]:
        """Fetch OHLCV across multiple timeframes for multi-timeframe analysis."""
        timeframes = {
            "15m": self.get_ohlcv(symbol, "15m", limit=100),
            "1h": self.get_ohlcv(symbol, "1h", limit=100),
            "4h": self.get_ohlcv(symbol, "4h", limit=100),
            "1d": self.get_ohlcv(symbol, "1d", limit=100),
        }
        return timeframes

    # ── Futures funding rates ──────────────────────────────────────────

    def get_funding_rate_history(
        self, symbol: str = "BTCUSDT", limit: int = 100, start_time: Optional[int] = None
    ) -> List[Dict]:
        """
        Fetch funding rate history for Binance perpetual futures.
        symbol: "BTCUSDT" (futures symbol)
        Returns list of dicts with keys: funding_time, funding_rate, symbol
        """
        try:
            params = {"symbol": symbol, "limit": limit}
            if start_time:
                params["startTime"] = start_time
            resp = self._session.get(
                f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "funding_time": int(r["fundingTime"]),
                    "funding_rate": float(r["fundingRate"]),
                    "symbol": r.get("symbol", symbol),
                }
                for r in data
            ]
        except requests.RequestException as e:
            logger.error("Binance funding rate request failed: %s", e)
            return []

    def get_current_funding_rate(self, symbol: str = "BTCUSDT") -> Optional[Dict]:
        """Get the current/next funding rate for a Binance perpetual swap."""
        try:
            resp = self._session.get(
                f"{BINANCE_FUTURES_BASE}/fapi/v1/premiumIndex",
                params={"symbol": symbol},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "symbol": data.get("symbol", symbol),
                "funding_rate": float(data.get("lastFundingRate", 0)),
                "funding_time": int(data.get("nextFundingTime", 0)),
                "mark_price": float(data.get("markPrice", 0)),
                "index_price": float(data.get("indexPrice", 0)),
                "last_price": float(data.get("lastPrice", 0)),
            }
        except requests.RequestException as e:
            logger.error("Binance current funding rate failed: %s", e)
            return None

    def get_ticker(self, symbol: str = "BTCUSDT") -> Optional[Dict]:
        """Get the current 24h ticker from Binance spot."""
        try:
            resp = self._session.get(
                f"{BINANCE_SPOT_BASE}/api/v3/ticker/24hr",
                params={"symbol": symbol},
                timeout=15,
            )
            resp.raise_for_status()
            t = resp.json()
            return {
                "last": float(t.get("lastPrice", 0)),
                "bid": float(t.get("bidPrice", 0)),
                "ask": float(t.get("askPrice", 0)),
                "volume_24h": float(t.get("volume", 0)),
                "quote_volume_24h": float(t.get("quoteVolume", 0)),
                "high_24h": float(t.get("highPrice", 0)),
                "low_24h": float(t.get("lowPrice", 0)),
                "open_24h": float(t.get("openPrice", 0)),
                "change_24h_pct": float(t.get("priceChangePercent", 0)),
            }
        except requests.RequestException as e:
            logger.error("Binance ticker request failed: %s", e)
            return None

    # ── Cross-exchange validation ──────────────────────────────────────

    def get_open_interest(self, symbol: str = "BTCUSDT") -> Optional[Dict]:
        """Get open interest for a futures pair — useful for sentiment signals."""
        try:
            resp = self._session.get(
                f"{BINANCE_FUTURES_BASE}/fapi/v1/openInterest",
                params={"symbol": symbol},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "symbol": symbol,
                "open_interest": float(data.get("openInterest", 0)),
            }
        except requests.RequestException as e:
            logger.error("Binance open interest request failed: %s", e)
            return None

    def get_long_short_ratio(
        self, symbol: str = "BTCUSDT", period: str = "1h", limit: int = 30
    ) -> List[Dict]:
        """Get top trader long/short ratio — a sentiment indicator."""
        try:
            resp = self._session.get(
                f"{BINANCE_FUTURES_BASE}/futures/data/topLongShortPositionRatio",
                params={"symbol": symbol, "period": period, "limit": limit},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "timestamp": int(r["timestamp"]),
                    "long_ratio": float(r["longAccount"]),
                    "short_ratio": float(r["shortAccount"]),
                    "long_short_ratio": float(r["longShortRatio"]),
                }
                for r in data
            ]
        except requests.RequestException as e:
            logger.error("Binance long/short ratio request failed: %s", e)
            return []
