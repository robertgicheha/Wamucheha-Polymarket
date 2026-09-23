"""
OKX connector. Dual role:
  1. Capital movement: buy USDC on OKX, withdraw to Polygon wallet for Polymarket funding
  2. Market data: OHLCV candles + funding rates for the crypto signal pipeline

No SDK needed: public endpoints are plain REST and private ones are signed
here with HMAC-SHA256 (OKX v5 auth).
"""
import base64
import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

from config.settings import settings

logger = logging.getLogger(__name__)


class OKXConnector:
    def __init__(self):
        self.api_key = settings.okx_api_key
        self.api_secret = settings.okx_api_secret
        self.api_passphrase = settings.okx_api_passphrase
        self._base_url = "https://www.okx.com"
        self._session = requests.Session()

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

    # ── Account / funding (authenticated, signed REST) ─────────────────
    #
    # Requirements on the OKX side:
    #   - API key with "Read" + "Withdraw" permission, IP-whitelisted to the
    #     bot host (OKX error 50110 = request IP not on the whitelist)
    #   - FUNDING_DEPOSIT_ADDRESS added to the withdrawal address book
    #   - USDC sitting in the FUNDING account (not the trading account)

    def _signed(self, method: str, path: str, body: Optional[Dict] = None) -> Dict:
        if not (self.api_key and self.api_secret and self.api_passphrase):
            raise RuntimeError("OKX API credentials are not configured")
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        payload = json.dumps(body, separators=(",", ":")) if body else ""
        prehash = f"{ts}{method}{path}{payload}"
        signature = base64.b64encode(
            hmac.new(self.api_secret.encode(), prehash.encode(), hashlib.sha256).digest()
        ).decode()
        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self.api_passphrase,
            "Content-Type": "application/json",
        }
        resp = self._session.request(
            method, f"{self._base_url}{path}", headers=headers, data=payload or None, timeout=15
        )
        data = resp.json()
        if data.get("code") != "0":
            raise OKXAPIError(data.get("code", "?"), data.get("msg", resp.text[:200]))
        return data

    def get_api_permissions(self) -> Dict:
        """Permissions and IP whitelist of the configured API key."""
        cfg = self._signed("GET", "/api/v5/account/config")["data"][0]
        return {"perm": cfg.get("perm", ""), "ip": cfg.get("ip", ""), "label": cfg.get("label", "")}

    def get_usdc_balance(self) -> float:
        """Available USDC in the OKX FUNDING account (withdrawals come from here)."""
        data = self._signed("GET", "/api/v5/asset/balances?ccy=USDC")["data"]
        return sum(float(d.get("availBal", 0) or 0) for d in data if d.get("ccy") == "USDC")

    def get_polygon_usdc_chain(self) -> Dict:
        """
        Resolve OKX's chain identifier for native USDC on Polygon (e.g.
        "USDC-Polygon") plus its withdrawal fee / minimum, instead of
        hard-coding a name OKX has changed before.
        """
        chains = self._signed("GET", "/api/v5/asset/currencies?ccy=USDC")["data"]
        candidates = [
            c for c in chains
            if "polygon" in c.get("chain", "").lower() and "bridged" not in c.get("chain", "").lower()
        ]
        if not candidates:
            raise RuntimeError(f"OKX offers no Polygon USDC chain: {[c.get('chain') for c in chains]}")
        c = candidates[0]
        return {
            "chain": c["chain"],
            "can_withdraw": bool(c.get("canWd")),
            "min_withdrawal": float(c.get("minWd", 0) or 0),
            "fee": float(c.get("fee") or c.get("minFee") or 0),
        }

    def withdraw_usdc_polygon(self, amount_usd: float, to_address: str) -> str:
        """
        Withdraw USDC on Polygon to `to_address` (must already be in the OKX
        withdrawal address book). Returns OKX's withdrawal id.
        """
        if settings.trading_mode != "live":
            raise RuntimeError("withdraw_usdc_polygon called while not in live mode")
        chain = self.get_polygon_usdc_chain()
        if not chain["can_withdraw"]:
            raise RuntimeError(f"OKX withdrawals on {chain['chain']} are currently suspended")
        if amount_usd < chain["min_withdrawal"]:
            raise RuntimeError(f"amount below OKX minimum {chain['min_withdrawal']}")
        body = {
            "ccy": "USDC",
            "amt": f"{amount_usd:.2f}",
            "dest": "4",  # 4 = on-chain withdrawal
            "toAddr": to_address,
            "chain": chain["chain"],
        }
        data = self._signed("POST", "/api/v5/asset/withdrawal", body)["data"]
        wd_id = data[0].get("wdId", "")
        logger.info("OKX withdrawal submitted: %s ($%.2f → %s)", wd_id, amount_usd, to_address)
        return wd_id

    # Backwards-compatible name
    def withdraw_to_polygon(self, amount_usd: float, to_address: str) -> str:
        return self.withdraw_usdc_polygon(amount_usd, to_address)

    def get_withdrawal_status(self, withdrawal_id: str) -> Optional[Dict]:
        """Status of a withdrawal (state: -3..2; 2 = success)."""
        data = self._signed("GET", f"/api/v5/asset/withdrawal-history?wdId={withdrawal_id}")["data"]
        return data[0] if data else None


class OKXAPIError(RuntimeError):
    def __init__(self, code: str, msg: str):
        super().__init__(f"OKX API error {code}: {msg}")
        self.code = code
