"""
trade_logger.py — Persistent trade + transaction logging to SQLite.

Every fill (entry, exit, size, side, PnL, fees) is written synchronously
immediately after execution — never batched — so a crash never loses a fill.

Also provides hourly rollup queries for the reporting system.

Tables:
  trades       — one row per fill (entry + exit combined)
  transactions — raw ledger of every money movement (stake, trade_cost,
                 pnl_credit, pnl_debit, fee, withdrawal, compound_in)
  daily_summary — end-of-day aggregation for fast dashboard queries
"""

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────
DB_PATH = os.environ.get("TRADE_LOG_DB_PATH", "data/trade_log.db")


def _ensure_dir():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


_ensure_dir()


# ── Schema ──────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id        TEXT UNIQUE NOT NULL,
    timestamp       TEXT NOT NULL,
    condition_id    TEXT NOT NULL,
    asset           TEXT NOT NULL,
    market_question TEXT DEFAULT '',

    -- Entry
    entry_side      TEXT NOT NULL,          -- YES or NO
    entry_price     REAL NOT NULL,
    entry_time      TEXT NOT NULL,
    size_usd        REAL NOT NULL,

    -- Exit (filled when position closes)
    exit_price      REAL,
    exit_time       TEXT,
    exit_reason     TEXT,                   -- resolution / take_profit / stop_loss / early_exit

    -- PnL
    pnl_usd         REAL DEFAULT 0.0,
    pnl_pct         REAL DEFAULT 0.0,
    fees_usd        REAL DEFAULT 0.0,
    won             INTEGER DEFAULT 0,     -- 1 = win, 0 = loss

    -- Metadata
    strategy        TEXT DEFAULT '',
    source          TEXT DEFAULT 'lifecycle',  -- lifecycle / arb / manual
    balance_at_entry REAL DEFAULT 0.0,       -- balance once the stake left (never overwritten)
    bankroll_after  REAL DEFAULT 0.0,        -- balance once the position settled
    metadata_json   TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS transactions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    tx_type         TEXT NOT NULL,          -- stake / trade_cost / pnl_credit / pnl_debit
                                        -- fee / withdrawal / compound_in / deposit
    amount_usd      REAL NOT NULL,
    balance_after   REAL NOT NULL,
    trade_id        TEXT,                   -- FK to trades.trade_id (nullable)
    description     TEXT DEFAULT '',
    metadata_json   TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS daily_summary (
    date            TEXT PRIMARY KEY,       -- YYYY-MM-DD
    total_trades    INTEGER DEFAULT 0,
    wins            INTEGER DEFAULT 0,
    losses          INTEGER DEFAULT 0,
    gross_profit    REAL DEFAULT 0.0,
    gross_loss      REAL DEFAULT 0.0,
    net_pnl         REAL DEFAULT 0.0,
    total_fees      REAL DEFAULT 0.0,
    total_volume    REAL DEFAULT 0.0,
    largest_win     REAL DEFAULT 0.0,
    largest_loss    REAL DEFAULT 0.0,
    avg_trade_size  REAL DEFAULT 0.0,
    win_rate_pct    REAL DEFAULT 0.0,
    bankroll_eod    REAL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_asset ON trades(asset);
CREATE INDEX IF NOT EXISTS idx_trades_won ON trades(won);
CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(tx_type);
CREATE INDEX IF NOT EXISTS idx_daily_summary_date ON daily_summary(date);
"""


def _empty_performance(since: Optional[str] = None) -> Dict[str, Any]:
    """Zeroed performance aggregate, shaped exactly like a populated one."""
    return {
        "since": since,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate_pct": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "net_pnl": 0.0,
        "total_fees": 0.0,
        "net_after_fees": 0.0,
        "total_volume": 0.0,
        "total_staked": 0.0,
        "total_returned": 0.0,
        "efficiency_pct": 0.0,
        "profit_factor": None,
        "largest_win": 0.0,
        "largest_loss": 0.0,
        "avg_trade_size": 0.0,
        "avg_pnl_per_trade": 0.0,
        "opening_balance": 0.0,
        "start_balance": 0.0,
        "end_balance": 0.0,
        "balance_delta": 0.0,
        "return_pct": 0.0,
        "by_asset": {},
        "by_side": {},
        "first_exit_time": None,
        "last_exit_time": None,
    }


class TradeLogger:
    """
    Thread-safe trade logger backed by SQLite.

    Usage:
        logger = TradeLogger()

        # Log a new trade entry
        trade_id = logger.log_entry(
            condition_id="0xabc...",
            asset="btc",
            side="YES",
            price=0.62,
            size_usd=50.0,
            strategy="kelly",
            bankroll_after=50.0,
        )

        # Log exit
        logger.log_exit(
            trade_id=trade_id,
            exit_price=0.78,
            exit_reason="take_profit",
            pnl_usd=12.80,
            fees_usd=0.92,
            bankroll_after=62.80,
        )

        # Get hourly rollup
        report = logger.get_hourly_report()
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or DB_PATH
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)
        logger.info("Trade logger initialized: %s", self._db_path)

    @staticmethod
    def _migrate(conn) -> None:
        """
        Additive column migrations.

        `CREATE TABLE IF NOT EXISTS` is a no-op on an already-created table,
        so new columns have to be added explicitly. Every step is idempotent
        and purely additive — nothing is dropped or retyped, so an existing
        trade history stays intact.
        """
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(trades)")}
        if "balance_at_entry" not in columns:
            conn.execute("ALTER TABLE trades ADD COLUMN balance_at_entry REAL DEFAULT 0.0")
            logger.info("Migrated trades table: added balance_at_entry")

    @contextmanager
    def _connect(self):
        """Thread-safe database connection."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Trade logging ──────────────────────────────────────────────────

    def log_entry(
        self,
        condition_id: str,
        asset: str,
        side: str,
        price: float,
        size_usd: float,
        strategy: str = "",
        source: str = "lifecycle",
        market_question: str = "",
        bankroll_after: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Log a trade entry. Returns the trade_id."""
        trade_id = f"trd_{int(time.time() * 1000)}_{asset}"
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO trades
                       (trade_id, timestamp, condition_id, asset, market_question,
                        entry_side, entry_price, entry_time, size_usd,
                        strategy, source, balance_at_entry, bankroll_after, metadata_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (trade_id, now, condition_id, asset, market_question,
                     side, price, now, size_usd,
                     strategy, source, bankroll_after, bankroll_after,
                     json.dumps(metadata or {})),
                )

                # Log transaction
                conn.execute(
                    """INSERT INTO transactions
                       (timestamp, tx_type, amount_usd, balance_after, trade_id, description)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (now, "trade_cost", -size_usd, bankroll_after, trade_id,
                     f"{side} {asset.upper()} @ {price:.3f} (${size_usd:.2f})"),
                )

        logger.info("TRADE LOG: entry %s %s @ %.3f ($%.2f) | %s", side, asset, price, size_usd, trade_id)
        return trade_id

    def log_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        pnl_usd: float,
        fees_usd: float = 0.0,
        bankroll_after: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Log a trade exit (close)."""
        now = datetime.now(timezone.utc).isoformat()
        won = 1 if pnl_usd > 0 else 0

        with self._lock:
            with self._connect() as conn:
                # Update the trade row
                conn.execute(
                    """UPDATE trades SET
                       exit_price=?, exit_time=?, exit_reason=?,
                       pnl_usd=?, fees_usd=?, won=?, bankroll_after=?
                       WHERE trade_id=?""",
                    (exit_price, now, exit_reason, pnl_usd, fees_usd, won,
                     bankroll_after, trade_id),
                )
                # `balance_at_entry` is deliberately left alone: it is the one
                # record of the balance before this trade's stake went out,
                # and reports need it to show a true before → after.

                # Calculate PnL percentage from entry
                row = conn.execute(
                    "SELECT size_usd, entry_price, metadata_json FROM trades WHERE trade_id=?",
                    (trade_id,),
                ).fetchone()
                if row and row["entry_price"] > 0:
                    pnl_pct = (pnl_usd / row["size_usd"]) * 100 if row["size_usd"] > 0 else 0
                    conn.execute(
                        "UPDATE trades SET pnl_pct=? WHERE trade_id=?",
                        (pnl_pct, trade_id),
                    )

                # Merge exit metadata into the entry's metadata_json rather
                # than overwriting it, so entry-time fields (e.g. the
                # predicted model_prob, for later Brier-score analysis)
                # survive alongside exit-time fields (e.g. final price).
                if metadata and row:
                    existing_meta = json.loads(row["metadata_json"] or "{}")
                    existing_meta.update(metadata)
                    conn.execute(
                        "UPDATE trades SET metadata_json=? WHERE trade_id=?",
                        (json.dumps(existing_meta), trade_id),
                    )

                # Log PnL transaction
                tx_type = "pnl_credit" if pnl_usd >= 0 else "pnl_debit"
                conn.execute(
                    """INSERT INTO transactions
                       (timestamp, tx_type, amount_usd, balance_after, trade_id, description)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (now, tx_type, pnl_usd, bankroll_after, trade_id,
                     f"PnL: ${pnl_usd:+.2f} ({exit_reason})"),
                )

                # Log fee if any
                if fees_usd > 0:
                    conn.execute(
                        """INSERT INTO transactions
                           (timestamp, tx_type, amount_usd, balance_after, trade_id, description)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (now, "fee", -fees_usd, bankroll_after, trade_id,
                         f"Trading fee: ${fees_usd:.4f}"),
                    )

        # Update daily summary
        self._update_daily_summary(pnl_usd, fees_usd, row["size_usd"] if row else 0, bankroll_after, won)

        emoji = "WIN" if won else "LOSS"
        logger.info(
            "TRADE LOG: exit %s %s | PnL: $%.4f | Fee: $%.4f | %s",
            trade_id, emoji, pnl_usd, fees_usd, exit_reason,
        )

    def log_transaction(
        self,
        tx_type: str,
        amount_usd: float,
        balance_after: float,
        description: str = "",
        trade_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Log a standalone transaction (deposit, withdrawal, compound, etc.)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO transactions
                       (timestamp, tx_type, amount_usd, balance_after, trade_id, description, metadata_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (now, tx_type, amount_usd, balance_after, trade_id,
                     description, json.dumps(metadata or {})),
                )

    # ── Daily summary ──────────────────────────────────────────────────

    def _update_daily_summary(
        self, pnl_usd: float, fees_usd: float, size_usd: float,
        bankroll_after: float, won: int,
    ):
        """Update the daily summary row for today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            with self._connect() as conn:
                existing = conn.execute(
                    "SELECT * FROM daily_summary WHERE date=?", (today,)
                ).fetchone()

                if existing:
                    total_trades = existing["total_trades"] + 1
                    wins = existing["wins"] + won
                    losses = existing["losses"] + (1 - won)
                    gross_profit = existing["gross_profit"] + (pnl_usd if pnl_usd > 0 else 0)
                    gross_loss = existing["gross_loss"] + (pnl_usd if pnl_usd < 0 else 0)
                    net_pnl = existing["net_pnl"] + pnl_usd
                    total_fees = existing["total_fees"] + fees_usd
                    total_volume = existing["total_volume"] + size_usd
                    largest_win = max(existing["largest_win"], pnl_usd if pnl_usd > 0 else 0)
                    largest_loss = min(existing["largest_loss"], pnl_usd if pnl_usd < 0 else 0)
                    avg_trade_size = total_volume / total_trades if total_trades > 0 else 0
                    win_rate_pct = (wins / total_trades * 100) if total_trades > 0 else 0

                    conn.execute(
                        """UPDATE daily_summary SET
                           total_trades=?, wins=?, losses=?, gross_profit=?, gross_loss=?,
                           net_pnl=?, total_fees=?, total_volume=?, largest_win=?,
                           largest_loss=?, avg_trade_size=?, win_rate_pct=?, bankroll_eod=?
                           WHERE date=?""",
                        (total_trades, wins, losses, gross_profit, gross_loss,
                         net_pnl, total_fees, total_volume, largest_win,
                         largest_loss, avg_trade_size, win_rate_pct, bankroll_after,
                         today),
                    )
                else:
                    conn.execute(
                        """INSERT INTO daily_summary
                           (date, total_trades, wins, losses, gross_profit, gross_loss,
                            net_pnl, total_fees, total_volume, largest_win, largest_loss,
                            avg_trade_size, win_rate_pct, bankroll_eod)
                           VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (today, won, 1 - won,
                         pnl_usd if pnl_usd > 0 else 0,
                         pnl_usd if pnl_usd < 0 else 0,
                         pnl_usd, fees_usd, size_usd,
                         pnl_usd if pnl_usd > 0 else 0,
                         pnl_usd if pnl_usd < 0 else 0,
                         size_usd, (100.0 if won else 0.0),
                         bankroll_after),
                    )

    # ── Report queries ─────────────────────────────────────────────────

    def get_hourly_report(self, hours: int = 1) -> Dict[str, Any]:
        """
        Get a comprehensive hourly rollup report.
        Returns dict with all stats needed for Telegram/Discord.
        """
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE entry_time >= ? AND exit_price IS NOT NULL",
                (since,),
            ).fetchall()

        if not rows:
            return {
                "period_hours": hours,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate_pct": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "net_pnl": 0.0,
                "total_fees": 0.0,
                "total_volume": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "avg_trade_size": 0.0,
                "avg_pnl_per_trade": 0.0,
                "current_bankroll": 0.0,
                "trades_by_asset": {},
            }

        total = len(rows)
        wins = sum(1 for r in rows if r["won"])
        losses = total - wins
        gross_profit = sum(r["pnl_usd"] for r in rows if r["pnl_usd"] > 0)
        gross_loss = sum(r["pnl_usd"] for r in rows if r["pnl_usd"] < 0)
        net_pnl = sum(r["pnl_usd"] for r in rows)
        total_fees = sum(r["fees_usd"] for r in rows if r["fees_usd"])
        total_volume = sum(r["size_usd"] for r in rows)
        pnls = [r["pnl_usd"] for r in rows]

        # Trades by asset
        by_asset: Dict[str, Dict] = {}
        for r in rows:
            asset = r["asset"]
            if asset not in by_asset:
                by_asset[asset] = {"trades": 0, "wins": 0, "pnl": 0.0}
            by_asset[asset]["trades"] += 1
            by_asset[asset]["wins"] += r["won"]
            by_asset[asset]["pnl"] += r["pnl_usd"]

        return {
            "period_hours": hours,
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round((wins / total * 100) if total > 0 else 0, 1),
            "gross_profit": round(gross_profit, 4),
            "gross_loss": round(gross_loss, 4),
            "net_pnl": round(net_pnl, 4),
            "total_fees": round(total_fees, 4),
            "total_volume": round(total_volume, 2),
            "largest_win": round(max(pnls) if pnls else 0, 4),
            "largest_loss": round(min(pnls) if pnls else 0, 4),
            "avg_trade_size": round(total_volume / total if total > 0 else 0, 2),
            "avg_pnl_per_trade": round(net_pnl / total if total > 0 else 0, 4),
            "current_bankroll": round(rows[-1]["bankroll_after"], 2) if rows else 0.0,
            "trades_by_asset": by_asset,
        }

    def get_session_summary(self) -> Dict[str, Any]:
        """Get stats for the entire current session (since bot started)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE exit_price IS NOT NULL ORDER BY entry_time ASC"
            ).fetchall()

        if not rows:
            return {"total_trades": 0, "session_pnl": 0.0, "session_duration_hours": 0}

        first_trade = rows[0]["entry_time"]
        last_trade = rows[-1]["exit_time"] or rows[-1]["entry_time"]

        total = len(rows)
        wins = sum(1 for r in rows if r["won"])
        net_pnl = sum(r["pnl_usd"] for r in rows)
        total_fees = sum(r["fees_usd"] for r in rows if r["fees_usd"])

        try:
            t_start = datetime.fromisoformat(first_trade)
            t_end = datetime.fromisoformat(last_trade)
            duration_hours = (t_end - t_start).total_seconds() / 3600
        except Exception:
            duration_hours = 0

        return {
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate_pct": round((wins / total * 100) if total > 0 else 0, 1),
            "session_pnl": round(net_pnl, 4),
            "session_fees": round(total_fees, 4),
            "session_duration_hours": round(duration_hours, 1),
            "current_bankroll": round(rows[-1]["bankroll_after"], 2),
        }

    def get_closed_trades_since(self, since: str, limit: int = 200) -> List[Dict]:
        """
        Closed trades (exit booked) at/after `since`, oldest first.

        `since` is an ISO-8601 UTC string — the same format written to the
        `exit_time` column, so plain string comparison is chronological.
        The reporting window is defined by exit time, not entry time: a trade
        entered at 12:04 and settled at 12:06 belongs to the window that
        closed at 12:07, which is when its PnL actually hit the bankroll.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM trades
                   WHERE exit_time IS NOT NULL AND exit_time >= ?
                   ORDER BY exit_time ASC LIMIT ?""",
                (since, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_transactions_since(self, since: str, limit: int = 500) -> List[Dict]:
        """Ledger movements at/after `since`, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM transactions
                   WHERE timestamp >= ? ORDER BY id ASC LIMIT ?""",
                (since, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_balance_at(self, at: str) -> Optional[float]:
        """
        Ledger balance as it stood at `at` (ISO-8601 UTC).

        Used as the "balance before" anchor for the first trade in a report
        window, so the digest can show a real before → after rather than
        guessing from the entry row.
        """
        with self._connect() as conn:
            row = conn.execute(
                """SELECT balance_after FROM transactions
                   WHERE timestamp <= ? ORDER BY id DESC LIMIT 1""",
                (at,),
            ).fetchone()
        return float(row["balance_after"]) if row and row["balance_after"] else None

    def get_performance(self, since: Optional[str] = None) -> Dict[str, Any]:
        """
        Full performance aggregate for a window (`since`) or the whole session.

        Beyond the usual W/L rollup this carries the numbers that make a
        report actionable: capital deployed, net ROI ("efficiency"), profit
        factor, and the balance the window started and ended on.
        """
        with self._connect() as conn:
            if since:
                rows = conn.execute(
                    """SELECT * FROM trades
                       WHERE exit_price IS NOT NULL AND exit_time >= ?
                       ORDER BY exit_time ASC""",
                    (since,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM trades
                       WHERE exit_price IS NOT NULL ORDER BY exit_time ASC"""
                ).fetchall()

        if not rows:
            return _empty_performance(since)

        total = len(rows)
        wins = sum(1 for r in rows if r["pnl_usd"] > 0)
        losses = total - wins
        gross_profit = sum(r["pnl_usd"] for r in rows if r["pnl_usd"] > 0)
        gross_loss = sum(r["pnl_usd"] for r in rows if r["pnl_usd"] < 0)
        net_pnl = sum(r["pnl_usd"] for r in rows)
        fees = sum(r["fees_usd"] or 0 for r in rows)
        volume = sum(r["size_usd"] or 0 for r in rows)
        pnls = [r["pnl_usd"] for r in rows]

        # Capital actually put at risk — the denominator for efficiency.
        staked = sum(r["size_usd"] or 0 for r in rows)
        # What came back to the balance: stake recovered plus/minus PnL.
        returned = staked + net_pnl

        by_asset: Dict[str, Dict] = {}
        by_side: Dict[str, Dict] = {}
        for r in rows:
            for bucket, key in ((by_asset, r["asset"]), (by_side, r["entry_side"])):
                if key not in bucket:
                    bucket[key] = {"trades": 0, "wins": 0, "pnl": 0.0}
                bucket[key]["trades"] += 1
                bucket[key]["wins"] += 1 if r["pnl_usd"] > 0 else 0
                bucket[key]["pnl"] += r["pnl_usd"]

        start_balance = rows[0]["bankroll_after"] or 0.0
        end_balance = rows[-1]["bankroll_after"] or 0.0
        # True pre-window balance: the ledger records the balance *after* the
        # first trade's stake went out, so add the stake back. Falling back to
        # `start - net_pnl` keeps pre-migration rows (no balance_at_entry)
        # producing a sensible number rather than nothing.
        first_entry_balance = rows[0]["balance_at_entry"] or 0.0
        opening_balance = (
            first_entry_balance + (rows[0]["size_usd"] or 0.0)
            if first_entry_balance
            else start_balance - net_pnl
        )

        return {
            "since": since,
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(wins / total * 100, 1),
            "gross_profit": round(gross_profit, 4),
            "gross_loss": round(gross_loss, 4),
            "net_pnl": round(net_pnl, 4),
            "total_fees": round(fees, 4),
            "net_after_fees": round(net_pnl - fees, 4),
            "total_volume": round(volume, 2),
            "total_staked": round(staked, 2),
            "total_returned": round(returned, 2),
            "efficiency_pct": round(net_pnl / staked * 100, 1) if staked > 0 else 0.0,
            "profit_factor": round(gross_profit / abs(gross_loss), 2) if gross_loss < 0 else None,
            "largest_win": round(max(pnls), 4),
            "largest_loss": round(min(pnls), 4),
            "avg_trade_size": round(volume / total, 2),
            "avg_pnl_per_trade": round(net_pnl / total, 4),
            "opening_balance": round(opening_balance, 2),
            "start_balance": round(start_balance, 2),
            "end_balance": round(end_balance, 2),
            "balance_delta": round(end_balance - opening_balance, 2),
            "return_pct": round((end_balance - opening_balance) / opening_balance * 100, 1)
            if opening_balance > 0 else 0.0,
            "by_asset": by_asset,
            "by_side": by_side,
            "first_exit_time": rows[0]["exit_time"],
            "last_exit_time": rows[-1]["exit_time"],
        }

    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get the most recent trades."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_daily_summaries(self, days: int = 7) -> List[Dict]:
        """Get daily summaries for the last N days."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM daily_summary ORDER BY date DESC LIMIT ?",
                (days,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_balance_history(self, limit: int = 100) -> List[Dict]:
        """Get balance over time from transactions."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT timestamp, balance_after FROM transactions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def backup(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of the database."""
        if backup_path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_dir = Path(self._db_path).parent / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = str(backup_dir / f"trade_log_{ts}.db")

        with self._connect() as conn:
            conn.execute(f"VACUUM INTO '{backup_path}'")

        logger.info("Database backup created: %s", backup_path)
        return backup_path
