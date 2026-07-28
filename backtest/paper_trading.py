"""
paper_trading.py

Drop-in paper-trading execution layer for the Polymarket bot.

Polymarket has no native demo mode, so this module fills that gap by:
  1. Pulling the REAL, live order book for a market from Polymarket's
     public CLOB API (no API key or wallet needed for reads).
  2. Simulating a fill by walking that book level-by-level, so your
     paper trades experience real slippage instead of a single
     mock price.
  3. Applying a fee model on the notional traded.
  4. Keeping a virtual ledger (cash balance + open positions + trade
     history) persisted to a local JSON file, so numbers survive
     bot restarts and can feed straight into your existing hourly /
     5-min reporting.

INTEGRATION
-----------
Wherever your bot currently does something like:

    result = clob_client.create_and_post_order(...)

swap it for:

    ledger = PaperLedger()
    result = ledger.place_order(token_id, side="BUY", usd_amount=25.0)

Everything else (your reporting, drawdown tracking, win-rate calc,
etc.) can stay the same -- just read balance / positions / trades
off `ledger` instead of off the real exchange client.

No external dependencies beyond `requests`.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests

logger = logging.getLogger(__name__)

CLOB_BOOK_URL = "https://clob.polymarket.com/book"


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Trade:
    id: str
    ts: float
    token_id: str
    side: str          # "BUY" or "SELL"
    shares: float
    avg_price: float
    notional: float
    fee: float
    slippage_vs_mid: float


@dataclass
class Position:
    token_id: str
    shares: float = 0.0
    cost_basis: float = 0.0   # total USD paid for currently-held shares

    @property
    def avg_entry_price(self) -> float:
        return self.cost_basis / self.shares if self.shares > 0 else 0.0


@dataclass
class LedgerState:
    balance: float
    positions: dict = field(default_factory=dict)   # token_id -> Position (as dict)
    trades: list = field(default_factory=list)       # list of Trade (as dict)


# --------------------------------------------------------------------------
# Order book helpers
# --------------------------------------------------------------------------

def fetch_order_book(token_id: str, timeout: float = 5.0) -> dict:
    """
    Fetch the live order book for a token from Polymarket's public CLOB API.
    Returns {"bids": [{"price": float, "size": float}, ...],
             "asks": [{"price": float, "size": float}, ...]}
    sorted best-price-first on each side.
    """
    try:
        resp = requests.get(CLOB_BOOK_URL, params={"token_id": token_id}, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json()

        def _norm(levels):
            out = [{"price": float(l["price"]), "size": float(l["size"])} for l in levels]
            return out

        bids = _norm(raw.get("bids", []))
        asks = _norm(raw.get("asks", []))
        bids.sort(key=lambda l: -l["price"])   # highest bid first
        asks.sort(key=lambda l: l["price"])    # lowest ask first
        return {"bids": bids, "asks": asks}
    except Exception as e:
        logger.error("Failed to fetch orderbook for %s: %s", token_id, e)
        return {"bids": [], "asks": []}


def walk_book(levels: list, usd_to_spend: float) -> tuple[float, float]:
    """
    Walk a sorted list of {"price", "size"} levels, spending usd_to_spend,
    consuming liquidity level by level.

    Returns (shares_filled, avg_price). If the book is too thin to fill
    the full amount, fills as much as available.
    """
    remaining = usd_to_spend
    shares = 0.0
    spent = 0.0

    for level in levels:
        price = level["price"]
        level_size_shares = level["size"]
        level_notional = price * level_size_shares

        if level_notional <= remaining:
            shares += level_size_shares
            spent += level_notional
            remaining -= level_notional
        else:
            # partial fill of this level
            partial_shares = remaining / price
            shares += partial_shares
            spent += remaining
            remaining = 0.0
            break

        if remaining <= 1e-9:
            break

    avg_price = (spent / shares) if shares > 0 else 0.0
    return shares, avg_price


def walk_book_by_shares(levels: list, shares_to_sell: float) -> tuple[float, float]:
    """
    Same as walk_book but for selling a fixed number of shares into bids.
    Returns (usd_received, avg_price).
    """
    remaining = shares_to_sell
    usd_received = 0.0
    filled_shares = 0.0

    for level in levels:
        price = level["price"]
        level_size_shares = level["size"]

        take = min(remaining, level_size_shares)
        usd_received += take * price
        filled_shares += take
        remaining -= take

        if remaining <= 1e-9:
            break

    avg_price = (usd_received / filled_shares) if filled_shares > 0 else 0.0
    return usd_received, avg_price


# --------------------------------------------------------------------------
# Fee model
# --------------------------------------------------------------------------

def compute_fee(price: float, shares: float, fee_bps: float) -> float:
    """
    fee = (fee_bps / 10000) * min(price, 1 - price) * shares

    This mirrors Polymarket's actual fee shape: fees are cheapest near the
    extremes (0.05 / 0.95) and most expensive near 0.50, since fee scales
    with min(price, 1-price) rather than flat notional.

    fee_bps is configurable because actual fee schedules vary by market
    and have changed over time -- set it to match whatever your bot
    currently assumes, or 0 if you want a fee-free baseline.
    """
    return (fee_bps / 10000.0) * min(price, 1.0 - price) * shares


# --------------------------------------------------------------------------
# The paper ledger
# --------------------------------------------------------------------------

class PaperLedger:
    def __init__(
        self,
        state_file: str = "data/paper_ledger_state.json",
        starting_balance: float = 1000.0,
        fee_bps: float = 200.0,
    ):
        self.state_file = state_file
        self.fee_bps = fee_bps
        self.state = self._load_or_init(starting_balance)

    # -- persistence ------------------------------------------------------

    def _load_or_init(self, starting_balance: float) -> LedgerState:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    raw = json.load(f)
                logger.info(
                    "Loaded paper ledger: balance=$%.2f, %d positions, %d trades",
                    raw["balance"],
                    len(raw.get("positions", {})),
                    len(raw.get("trades", [])),
                )
                return LedgerState(
                    balance=raw["balance"],
                    positions=raw.get("positions", {}),
                    trades=raw.get("trades", []),
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Corrupt paper ledger state, reinitializing: %s", e)
        logger.info("Initializing paper ledger with $%.2f balance", starting_balance)
        return LedgerState(balance=starting_balance)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(asdict(self.state), f, indent=2)

    # -- core actions -------------------------------------------------------

    def place_order(self, token_id: str, side: str, usd_amount: Optional[float] = None,
                     shares: Optional[float] = None) -> dict:
        """
        Simulate an order fill against the live book.

        BUY:  pass usd_amount (how much cash to spend).
        SELL: pass shares (how many shares of your position to sell).

        Returns a dict summarizing the simulated fill, shaped similarly
        to what a real exchange client would hand back, so your existing
        bot logic (logging, reporting) needs minimal changes.
        """
        side = side.upper()
        book = fetch_order_book(token_id)

        if side == "BUY":
            if not usd_amount or usd_amount <= 0:
                raise ValueError("usd_amount required and must be > 0 for BUY")
            if usd_amount > self.state.balance:
                return {"status": "rejected", "reason": "insufficient paper balance"}

            asks = book["asks"]
            if not asks:
                return {"status": "rejected", "reason": "no liquidity (empty ask book)"}

            mid = (asks[0]["price"] + book["bids"][0]["price"]) / 2 if book["bids"] else asks[0]["price"]
            filled_shares, avg_price = walk_book(asks, usd_amount)
            if filled_shares == 0:
                return {"status": "rejected", "reason": "no fill"}

            notional = filled_shares * avg_price
            fee = compute_fee(avg_price, filled_shares, self.fee_bps)
            total_cost = notional + fee

            if total_cost > self.state.balance:
                # trim fill to what balance can actually cover incl. fee
                scale = self.state.balance / total_cost
                filled_shares *= scale
                notional *= scale
                fee *= scale
                total_cost = notional + fee

            self.state.balance -= total_cost

            pos = self.state.positions.get(token_id, {"token_id": token_id, "shares": 0.0, "cost_basis": 0.0})
            pos["shares"] += filled_shares
            pos["cost_basis"] += notional
            self.state.positions[token_id] = pos

            slippage = avg_price - mid
            trade = Trade(
                id=str(uuid.uuid4())[:8], ts=time.time(), token_id=token_id, side="BUY",
                shares=filled_shares, avg_price=avg_price, notional=notional,
                fee=fee, slippage_vs_mid=slippage,
            )
            self.state.trades.append(asdict(trade))
            self._save()
            logger.info(
                "PAPER BUY %s: %.4f shares @ avg $%.4f (notional=$%.4f, fee=$%.4f, slippage=%.4f)",
                token_id, filled_shares, avg_price, notional, fee, slippage,
            )
            return {"status": "filled", **asdict(trade), "balance_after": self.state.balance}

        elif side == "SELL":
            if not shares or shares <= 0:
                raise ValueError("shares required and must be > 0 for SELL")
            pos = self.state.positions.get(token_id)
            if not pos or pos["shares"] < shares:
                return {"status": "rejected", "reason": "insufficient paper position"}

            bids = book["bids"]
            if not bids:
                return {"status": "rejected", "reason": "no liquidity (empty bid book)"}

            mid = (bids[0]["price"] + book["asks"][0]["price"]) / 2 if book["asks"] else bids[0]["price"]
            usd_received, avg_price = walk_book_by_shares(bids, shares)
            if usd_received == 0:
                return {"status": "rejected", "reason": "no fill"}

            fee = compute_fee(avg_price, shares, self.fee_bps)
            net_proceeds = usd_received - fee

            # reduce position proportionally (average cost basis method)
            avg_entry = pos["cost_basis"] / pos["shares"]
            cost_of_sold = avg_entry * shares
            pos["shares"] -= shares
            pos["cost_basis"] -= cost_of_sold
            if pos["shares"] <= 1e-9:
                pos["shares"] = 0.0
                pos["cost_basis"] = 0.0
            self.state.positions[token_id] = pos

            self.state.balance += net_proceeds

            slippage = mid - avg_price
            trade = Trade(
                id=str(uuid.uuid4())[:8], ts=time.time(), token_id=token_id, side="SELL",
                shares=shares, avg_price=avg_price, notional=usd_received,
                fee=fee, slippage_vs_mid=slippage,
            )
            self.state.trades.append(asdict(trade))
            self._save()
            realized_pnl = usd_received - cost_of_sold - fee
            logger.info(
                "PAPER SELL %s: %.4f shares @ avg $%.4f (proceeds=$%.4f, fee=$%.4f, PnL=$%.4f)",
                token_id, shares, avg_price, net_proceeds, fee, realized_pnl,
            )
            return {"status": "filled", **asdict(trade), "balance_after": self.state.balance,
                    "realized_pnl": realized_pnl}

        else:
            raise ValueError("side must be 'BUY' or 'SELL'")

    # -- reporting ----------------------------------------------------------

    def summary(self) -> dict:
        trades = self.state.trades
        total_fees = sum(t["fee"] for t in trades)
        return {
            "balance": round(self.state.balance, 4),
            "open_positions": {k: v for k, v in self.state.positions.items() if v["shares"] > 0},
            "total_trades": len(trades),
            "total_fees": round(total_fees, 4),
        }
