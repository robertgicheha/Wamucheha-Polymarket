"""
Order execution for the up/down lifecycle engine.

Both executors take the same inputs and return the same `Fill`, so paper
results are directly comparable to live ones:

  PaperExecutor — fills taker orders against the REAL order book levels
                  (walking depth up to the price bound) and charges the real
                  taker fee. No slippage fudge factor.
  LiveExecutor  — sends FAK (fill-and-kill) taker orders through the
                  polymarket-client SDK with a hard price bound, then reads
                  the wallet's actual token balance to learn what filled.

Every live entry passes these guards first; any failure blocks the order:
  - TRADING_MODE=live and LIVE_TRADING_ACK set to the exact phrase
  - no KILL_SWITCH file present (touch it to stop new entries instantly;
    exits and redemptions keep working)
  - order <= LIVE_MAX_ORDER_USD and open exposure <= LIVE_MAX_OPEN_EXPOSURE_USD
  - enough pUSD in the wallet
"""
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from config.settings import LIVE_TRADING_ACK_PHRASE, settings
from strategies.fair_value import taker_fee_per_share

logger = logging.getLogger(__name__)


@dataclass
class Fill:
    side: str            # "BUY" / "SELL"
    token_id: str
    shares: float        # shares actually bought / sold
    avg_price: float     # average execution price (before fees)
    fees_usd: float
    cash_usd: float      # BUY: total paid incl. fees; SELL: net proceeds after fees
    order_id: str = ""
    simulated: bool = False


class BaseExecutor:
    live = False

    def buy(self, token_id: str, usd_amount: float, max_price: float,
            asks: List[Tuple[float, float]], fee_rate: float, fee_exponent: float) -> Optional[Fill]:
        raise NotImplementedError

    def sell(self, token_id: str, shares: float, min_price: float,
             bids: List[Tuple[float, float]], fee_rate: float, fee_exponent: float) -> Optional[Fill]:
        raise NotImplementedError

    def entries_allowed(self) -> Tuple[bool, str]:
        if os.path.exists(settings.kill_switch_file):
            return False, f"kill switch file '{settings.kill_switch_file}' present"
        return True, ""

    def collateral_balance(self) -> Optional[float]:
        return None

    def redeem(self, condition_id: str) -> bool:
        return True

    def redeem_all(self) -> int:
        return 0

    def cancel_all(self) -> None:
        pass


def simulate_taker(
    side: str,
    levels: List[Tuple[float, float]],
    bound: float,
    fee_rate: float,
    fee_exponent: float,
    usd_amount: float = 0.0,
    shares: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Walk best-first book levels. BUY spends up to `usd_amount` (incl. fees) on
    asks priced <= bound; SELL sells up to `shares` into bids priced >= bound.
    Returns (shares, notional_before_fees, fees).
    """
    filled = notional = fees = 0.0
    for price, size in levels:
        if side == "BUY" and price > bound + 1e-9:
            break
        if side == "SELL" and price < bound - 1e-9:
            break
        fee = taker_fee_per_share(price, fee_rate, fee_exponent)
        if side == "BUY":
            budget = usd_amount - notional - fees
            take = min(size, budget / (price + fee)) if price + fee > 0 else 0.0
        else:
            take = min(size, shares - filled)
        if take <= 1e-9:
            break
        filled += take
        notional += take * price
        fees += take * fee
    return filled, notional, fees


class PaperExecutor(BaseExecutor):
    live = False

    def buy(self, token_id, usd_amount, max_price, asks, fee_rate, fee_exponent):
        shares, notional, fees = simulate_taker(
            "BUY", asks, max_price, fee_rate, fee_exponent, usd_amount=usd_amount,
        )
        if shares <= 0:
            return None
        return Fill("BUY", token_id, shares, notional / shares, fees, notional + fees,
                    order_id=f"paper_{int(time.time() * 1000)}", simulated=True)

    def sell(self, token_id, shares, min_price, bids, fee_rate, fee_exponent):
        sold, notional, fees = simulate_taker(
            "SELL", bids, min_price, fee_rate, fee_exponent, shares=shares,
        )
        if sold <= 0:
            return None
        return Fill("SELL", token_id, sold, notional / sold, fees, notional - fees,
                    order_id=f"paper_{int(time.time() * 1000)}", simulated=True)


class LiveExecutor(BaseExecutor):
    live = True

    def __init__(self, connector, exposure_fn=None):
        """
        connector: PolymarketConnector
        exposure_fn: callable returning current open exposure in USD (cost basis)
        """
        self.connector = connector
        self.exposure_fn = exposure_fn or (lambda: 0.0)
        self._lock = threading.Lock()

    def entries_allowed(self) -> Tuple[bool, str]:
        if settings.trading_mode != "live":
            return False, "TRADING_MODE is not live"
        if settings.live_trading_ack != LIVE_TRADING_ACK_PHRASE:
            return False, "LIVE_TRADING_ACK not set"
        return super().entries_allowed()

    def collateral_balance(self) -> Optional[float]:
        try:
            return self.connector.get_collateral_balance()
        except Exception as e:
            logger.warning("Collateral balance read failed: %s", e)
            return None

    def _token_balance(self, token_id: str) -> Optional[float]:
        try:
            return self.connector.get_token_balance(token_id)
        except Exception as e:
            logger.warning("Token balance read failed for %s: %s", token_id, e)
            return None

    def buy(self, token_id, usd_amount, max_price, asks, fee_rate, fee_exponent):
        ok, reason = self.entries_allowed()
        if not ok:
            logger.warning("Live BUY blocked: %s", reason)
            return None
        with self._lock:
            if usd_amount > settings.live_max_order_usd:
                logger.info("Capping live order $%.2f → $%.2f (LIVE_MAX_ORDER_USD)",
                            usd_amount, settings.live_max_order_usd)
                usd_amount = settings.live_max_order_usd
            exposure = self.exposure_fn()
            if exposure + usd_amount > settings.live_max_open_exposure_usd:
                logger.warning("Live BUY blocked: exposure $%.2f + $%.2f > cap $%.2f",
                               exposure, usd_amount, settings.live_max_open_exposure_usd)
                return None
            balance = self.collateral_balance()
            if balance is None or balance < usd_amount:
                logger.warning("Live BUY blocked: pUSD balance %s < $%.2f", balance, usd_amount)
                return None

            before = self._token_balance(token_id) or 0.0
            try:
                resp = self.connector.buy_fak(token_id, usd_amount, max_price)
            except Exception as e:
                logger.error("Live BUY failed: %s", e)
                return None
            if not getattr(resp, "ok", False):
                logger.info("Live BUY not filled: %s %s", getattr(resp, "code", "?"),
                            getattr(resp, "message", ""))
                return None

            spent = float(resp.making_amount)
            shares = float(resp.taking_amount)
            # The wallet balance is the source of truth for what we hold
            # (fees may be taken in shares); retry briefly while it settles.
            for _ in range(5):
                after = self._token_balance(token_id)
                if after is not None and after > before:
                    shares = after - before
                    break
                time.sleep(1)
            if shares <= 0 or spent <= 0:
                logger.warning("Live BUY accepted but no fill detected (order %s)", resp.order_id)
                return None
            avg = spent / shares
            fees = max(0.0, shares * taker_fee_per_share(min(avg, 0.99), fee_rate, fee_exponent))
            logger.info("LIVE BUY filled: %.2f sh of %s for $%.2f (order %s)",
                        shares, token_id[:10], spent, resp.order_id)
            return Fill("BUY", token_id, shares, avg, fees, spent, order_id=str(resp.order_id))

    def sell(self, token_id, shares, min_price, bids, fee_rate, fee_exponent):
        with self._lock:
            held = self._token_balance(token_id)
            if held is not None:
                shares = min(shares, held)
            if shares <= 0:
                return None
            try:
                resp = self.connector.sell_fak(token_id, shares, min_price)
            except Exception as e:
                logger.error("Live SELL failed: %s", e)
                return None
            if not getattr(resp, "ok", False):
                logger.info("Live SELL not filled: %s %s", getattr(resp, "code", "?"),
                            getattr(resp, "message", ""))
                return None
            sold = float(resp.making_amount)
            proceeds = float(resp.taking_amount)
            if sold <= 0:
                return None
            avg = proceeds / sold
            fees = sold * taker_fee_per_share(avg, fee_rate, fee_exponent)
            logger.info("LIVE SELL filled: %.2f sh of %s for $%.2f (order %s)",
                        sold, token_id[:10], proceeds, resp.order_id)
            return Fill("SELL", token_id, sold, avg, fees, proceeds, order_id=str(resp.order_id))

    def redeem(self, condition_id: str) -> bool:
        try:
            self.connector.redeem(condition_id)
            logger.info("Redeemed positions for %s", condition_id)
            return True
        except Exception as e:
            logger.warning("Redeem failed for %s: %s", condition_id, e)
            return False

    def redeem_all(self) -> int:
        """Redeem every resolved position the wallet still holds."""
        try:
            ids = self.connector.list_redeemable_condition_ids()
        except Exception as e:
            logger.warning("Could not list redeemable positions: %s", e)
            return 0
        return sum(1 for cid in ids if self.redeem(cid))

    def cancel_all(self) -> None:
        self.connector.cancel_all()
