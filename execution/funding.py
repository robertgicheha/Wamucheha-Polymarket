"""
Moves USDC from OKX or Binance to the Polymarket account.

Flow: exchange --(USDC on Polygon)--> Polymarket bridge deposit address
      --(auto-wrapped by Polymarket)--> pUSD in your trading wallet.

The exchanges are funding SOURCES, never trading wallets and never the CLOB
"funder". Safety rules enforced here:
  - funds only ever go to FUNDING_DEPOSIT_ADDRESS, and that address must be
    the EVM bridge address Polymarket issues for your trading wallet
  - per-withdrawal cap (FUNDING_MAX_WITHDRAWAL_USD) and rolling 24h cap
    (FUNDING_MAX_DAILY_USD), persisted to disk so restarts can't reset it
  - one withdrawal in flight at a time (FUNDING_COOLDOWN_SECONDS)
"""
import json
import logging
import os
import time
from typing import Dict, List, Optional

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

BRIDGE_API = "https://bridge.polymarket.com"
FUNDING_LEDGER = os.path.join("data", "funding_ledger.json")
FUNDING_COOLDOWN_SECONDS = 30 * 60
DAY_SECONDS = 86400


def get_bridge_evm_address(wallet: str) -> Optional[str]:
    """Polymarket's EVM deposit (bridge) address for a trading wallet."""
    resp = requests.post(f"{BRIDGE_API}/deposit", json={"address": wallet}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    address = (data.get("address") or data).get("evm") if isinstance(data, dict) else None
    return address


class FundingManager:
    def __init__(self, okx=None, binance=None, ledger_path: str = FUNDING_LEDGER):
        self.sources = {}
        if okx is not None:
            self.sources["okx"] = okx
        if binance is not None:
            self.sources["binance"] = binance
        self.ledger_path = ledger_path
        self._ledger: List[Dict] = self._load()

    # ── ledger ────────────────────────────────────────────────────────

    def _load(self) -> List[Dict]:
        try:
            with open(self.ledger_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.ledger_path) or ".", exist_ok=True)
        tmp = self.ledger_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._ledger, f, indent=2)
        os.replace(tmp, self.ledger_path)

    def withdrawn_last_24h(self, now: Optional[float] = None) -> float:
        now = now or time.time()
        return sum(e["amount"] for e in self._ledger if now - e["ts"] < DAY_SECONDS)

    def last_withdrawal_ts(self) -> float:
        return max((e["ts"] for e in self._ledger), default=0.0)

    # ── checks ────────────────────────────────────────────────────────

    def verify_destination(self, trading_wallet: str) -> None:
        """
        Refuse to fund anything but the bridge address Polymarket issues for
        our own trading wallet — a typo'd or stale address loses the money.
        """
        dest = settings.funding_deposit_address
        if not dest:
            raise RuntimeError("FUNDING_DEPOSIT_ADDRESS is not set")
        expected = get_bridge_evm_address(trading_wallet)
        if not expected or expected.lower() != dest.lower():
            raise RuntimeError(
                f"FUNDING_DEPOSIT_ADDRESS {dest} is not the Polymarket bridge address "
                f"for wallet {trading_wallet} (expected {expected})"
            )

    def check_amount(self, amount: float, now: Optional[float] = None) -> None:
        now = now or time.time()
        if amount <= 0:
            raise ValueError("amount must be positive")
        if amount < 2:
            raise ValueError("Polymarket ignores deposits under $2")
        if amount > settings.funding_max_withdrawal_usd:
            raise ValueError(
                f"${amount:.2f} exceeds FUNDING_MAX_WITHDRAWAL_USD "
                f"(${settings.funding_max_withdrawal_usd:.2f})"
            )
        used = self.withdrawn_last_24h(now)
        if used + amount > settings.funding_max_daily_usd:
            raise ValueError(
                f"24h funding cap: ${used:.2f} already sent, cap "
                f"${settings.funding_max_daily_usd:.2f}"
            )
        since = now - self.last_withdrawal_ts()
        if since < FUNDING_COOLDOWN_SECONDS:
            raise ValueError(f"previous withdrawal {since:.0f}s ago — cooldown active")

    # ── actions ───────────────────────────────────────────────────────

    def fund(self, source: str, amount: float, trading_wallet: str) -> str:
        if settings.trading_mode != "live":
            raise RuntimeError("funding is only possible with TRADING_MODE=live")
        client = self.sources.get(source)
        if client is None:
            raise ValueError(f"unknown or unconfigured funding source '{source}'")
        self.check_amount(amount)
        self.verify_destination(trading_wallet)
        available = client.get_usdc_balance()
        if available < amount:
            raise RuntimeError(f"{source} has only ${available:.2f} USDC available")

        wd_id = client.withdraw_usdc_polygon(amount, settings.funding_deposit_address)
        self._ledger.append({
            "ts": time.time(), "source": source, "amount": amount,
            "id": wd_id, "to": settings.funding_deposit_address,
        })
        self._save()
        return wd_id

    def maybe_auto_fund(self, collateral_balance: Optional[float], trading_wallet: str) -> Optional[str]:
        """Top up from FUNDING_SOURCE when the wallet runs low (opt-in)."""
        if not settings.auto_funding_enabled or collateral_balance is None:
            return None
        if collateral_balance >= settings.funding_min_balance_usd:
            return None
        try:
            return self.fund(settings.funding_source, settings.funding_topup_usd, trading_wallet)
        except Exception as e:
            logger.warning("Auto-funding skipped: %s", e)
            return None
