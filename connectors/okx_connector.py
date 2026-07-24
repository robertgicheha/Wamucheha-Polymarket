"""
OKX connector. Dual role:
  1. Capital movement: buy USDC on OKX, withdraw to Polygon wallet for Polymarket funding
  2. Market data: OHLCV candles + funding rates for the crypto signal pipeline

Install: pip install python-okx
"""
import logging
import time
from typing import Dict, List, Optional, Tuple

import requests

from config.settings import settings

logger = logging.getLogger(__name__)


class OKXConnector:
    def __init__(self):
        self.api_key = settings.okx_api_key
        self.api_secret = settings.okx_api_secret
        self.api_passphrase = settings.okx_api_passphrase
        self._client = None
        self._base_url = "https://www.okx.com"
        self._session = requests.Session()

    def _get_client(self):
        """Lazily initialize the OKX SDK client (for authenticated endpoints)."""
        if self._client is None:
            from okx.market_data import MarketAPI
            from okx.account import AccountAPI
            self._client = {
            "account": AccountAPI(
                self.api_key, self.api_secret, self.api_passphrase, False, "1"
            ),
            "market": MarketAPI(
                self.api_key, self.api_secret, self.api_passphrase, False, "1"
            ),
        }
        return self._client

    # ── Market data (public, no auth needed) ──────────────────────────

    def get_ohlcv(
        self,
        symbol: str = "BTC-USDT",
        bar: str = "1H",
        limit: int = 100,
        after: Optional[int] = None,
        before: Optional[int] = None,
    ) -> List[Dict]:
        """
        Fetch OHLCV candles from OKX.
        symbol: e.g. "BTC-USDT", "ETH-USDT"
        bar: "1m","5m","15m","1H","4H","1D","1W"
        Returns list of dicts with keys: ts, open, high, low, close, vol, volCcy
        """
        try:
            params = {"instId": symbol, "bar": bar, "limit": str(limit)}
            if after:
                params["after"] = str(after)
            if before:
                params["before"] = str(before)
            resp = self._session.get(
                f"{self._base_url}/api/v5/market/candles",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != "0":
                logger.error("OKX OHLCV error: %s", data.get("msg"))
                return []
            candles = []
            for c in data.get("data", []):
                candles.append({
                    "ts": int(c[0]),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "vol": float(c[5]),
                    "vol_ccy": float(c[6]),
                    "vol_ccy_quote": float(c[7]) if len(c) > 7 else 0,
                    "confirm": c[8] if len(c) > 8 else "0",
                })
            return candles
        except requests.RequestException as e:
            logger.error("OKX OHLCV request failed: %s", e)
            return []

    def get_funding_rate_history(
        self, symbol: str = "BTC-USDT-SWAP", limit: int = 100
    ) -> List[Dict]:
        """
        Fetch funding rate history for perpetual swap contracts.
        symbol: "BTC-USDT-SWAP", "ETH-USDT-SWAP"
        Returns list of dicts with keys: fundingTime, fundingRate, symbol
        """
        try:
            resp = self._session.get(
                f"{self._base_url}/api/v5/public/funding-rate-history",
                params={"instId": symbol, "limit": str(limit)},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != "0":
                logger.error("OKX funding rate error: %s", data.get("msg"))
                return []
            return [
                {
                    "funding_time": int(r["fundingTime"]),
                    "funding_rate": float(r["fundingRate"]),
                    "symbol": r.get("instId", symbol),
                }
                for r in data.get("data", [])
            ]
        except requests.RequestException as e:
            logger.error("OKX funding rate request failed: %s", e)
            return []

    def get_current_funding_rate(self, symbol: str = "BTC-USDT-SWAP") -> Optional[Dict]:
        """Get the current/next funding rate for a perpetual swap."""
        try:
            resp = self._session.get(
                f"{self._base_url}/api/v5/public/funding-rate",
                params={"instId": symbol},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != "0" or not data.get("data"):
                return None
            r = data["data"][0]
            return {
                "funding_time": int(r.get("fundingTime", 0)),
                "funding_rate": float(r.get("fundingRate", 0)),
                "next_funding_time": int(r.get("nextFundingTime", 0)),
                "symbol": r.get("instId", symbol),
            }
        except requests.RequestException as e:
            logger.error("OKX current funding rate failed: %s", e)
            return None

    def get_ticker(self, symbol: str = "BTC-USDT") -> Optional[Dict]:
        """Get the current ticker (last price, 24h volume, etc.)."""
        try:
            resp = self._session.get(
                f"{self._base_url}/api/v5/market/ticker",
                params={"instId": symbol},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != "0" or not data.get("data"):
                return None
            t = data["data"][0]
            return {
                "last": float(t.get("last", 0)),
                "bid": float(t.get("bidPx", 0)),
                "ask": float(t.get("askPx", 0)),
                "volume_24h": float(t.get("vol24h", 0)),
                "volume_ccy_24h": float(t.get("volCcy24h", 0)),
                "high_24h": float(t.get("high24h", 0)),
                "low_24h": float(t.get("low24h", 0)),
                "open_24h": float(t.get("open24h", 0)),
                "change_24h_pct": (
                    (float(t.get("last", 0)) - float(t.get("open24h", 0)))
                    / float(t.get("open24h", 1))
                    * 100
                    if float(t.get("open24h", 0)) > 0
                    else 0
                ),
            }
        except requests.RequestException as e:
            logger.error("OKX ticker request failed: %s", e)
            return None

    # ── Account / funding (authenticated) ──────────────────────────────

    def get_usdc_balance(self) -> float:
        """Query OKX account balance for USDC."""
        try:
            client = self._get_client()
            result = client["account"].get_account_balance()
            for detail in result.data[0].details:
                if detail.ccy == "USDC":
                    return float(detail.availBal)
            return 0.0
        except Exception as e:
            logger.error("OKX balance query failed: %s", e)
            return 0.0

    def withdraw_to_polygon(self, amount_usd: float, to_address: str) -> str:
        """
        Submit a USDC withdrawal from OKX to a Polygon-network address.
        This is the funding step: OKX -> your Polymarket-linked wallet.
        Returns withdrawal ID for tracking.
        """
        if settings.trading_mode != "live":
            raise RuntimeError("withdraw_to_polygon called while not in live mode")
        try:
            client = self._get_client()
            result = client["account"].withdraw(
                ccy="USDC",
                amt=str(amount_usd),
                dest="4",  # 4 = on-chain address
                toAddr=to_address,
                chain="Polygon",
            )
            if result.code == "0" and result.data:
                withdrawal_id = result.data[0].get("wdId", "")
                logger.info("OKX withdrawal submitted: %s", withdrawal_id)
                return withdrawal_id
            raise RuntimeError(f"OKX withdrawal failed: {result.msg}")
        except Exception as e:
            logger.error("OKX withdrawal failed: %s", e)
            raise

    def get_withdrawal_status(self, withdrawal_id: str) -> Optional[Dict]:
        """Check the status of a pending withdrawal."""
        try:
            client = self._get_client()
            result = client["account"].get_deposit_withdraw_history(
                wdId=withdrawal_id
            )
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error("OKX withdrawal status query failed: %s", e)
            return None
