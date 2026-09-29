"""
scheduler.py — Periodic trade reporting.

Runs three jobs:
  1. Every 5 minutes: a trade digest covering only what closed in the window.
  2. Every hour: a full rollup of the hour's trades plus session context.
  3. On demand: the same digest, for explicit operator requests.

All three are *conditional*: a window in which nothing closed and no money
moved sends nothing at all. A bot that pings "still here" every five minutes
trains you to ignore it — the report only carries information when something
actually happened.

Rendering lives in alerts.trade_digest; this module owns the timing, the
window boundaries and the decision to stay silent.
"""

import logging
import time
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from alerts.trade_digest import (
    build_digest,
    format_forced_digest,
    format_hourly_report,
)

logger = logging.getLogger(__name__)


class BotScheduler:
    """
    Lightweight scheduler that runs periodic reporting in a background thread.

    Usage:
        scheduler = BotScheduler(trade_logger=logger, notifier=notifier)
        scheduler.start()
        # ... later ...
        scheduler.stop()
    """

    def __init__(
        self,
        trade_logger,
        notifier,
        status_interval_seconds: int = 300,   # 5 minutes
        report_interval_seconds: int = 3600,   # 1 hour
        get_risk_summary: Optional[Callable] = None,
        get_arb_stats: Optional[Callable] = None,
        get_lifecycle_stats: Optional[Callable] = None,
        mode: str = "PAPER",
        max_trades_per_report: int = 6,
    ):
        self.trade_logger = trade_logger
        self.notifier = notifier
        self.status_interval = status_interval_seconds
        self.report_interval = report_interval_seconds
        self.get_risk_summary = get_risk_summary
        self.get_arb_stats = get_arb_stats
        self.get_lifecycle_stats = get_lifecycle_stats
        self.mode = mode
        self.max_trades_per_report = max_trades_per_report

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_status_time = 0.0
        self._last_report_time = 0.0
        self._start_time = time.time()
        self._last_digest_count = 0

    def start(self):
        """Start the scheduler in a background thread."""
        if self._running:
            return
        self._running = True
        now = time.time()
        self._last_status_time = now
        self._last_report_time = now
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="bot-scheduler")
        self._thread.start()
        logger.info(
            "Scheduler started: %s, %s (silent when no trades)",
            f"digest every {self.status_interval}s" if self.status_interval > 0 else "digest off",
            f"rollup every {self.report_interval}s" if self.report_interval > 0 else "rollup off",
        )

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scheduler stopped")

    def _run_loop(self):
        """Main scheduler loop."""
        while self._running:
            now = time.time()

            if self.status_interval > 0 and now - self._last_status_time >= self.status_interval:
                window_start = self._since(self._last_status_time)
                try:
                    self._send_trade_digest(window_start, window_end=now)
                except Exception as e:
                    logger.error("Trade digest failed: %s", e)
                self._last_status_time = now

            if self.report_interval > 0 and now - self._last_report_time >= self.report_interval:
                window_start = self._since(self._last_report_time)
                try:
                    self._send_hourly_report(window_start, window_end=now)
                except Exception as e:
                    logger.error("Hourly report failed: %s", e)
                self._last_report_time = now

            time.sleep(10)  # check every 10 seconds

    # ── Window helpers ─────────────────────────────────────────────────

    def _since(self, epoch: float) -> str:
        """Epoch seconds → the ISO-8601 UTC string the ledger is keyed on."""
        return datetime.fromtimestamp(epoch, timezone.utc).isoformat()

    def _window_bounds(self, epoch_start: float, epoch_end: float):
        return (
            datetime.fromtimestamp(epoch_start, timezone.utc),
            datetime.fromtimestamp(epoch_end, timezone.utc),
        )

    def _uptime_hours(self) -> float:
        return (time.time() - self._start_time) / 3600

    def _risk(self) -> Dict[str, Any]:
        if not self.get_risk_summary:
            return {}
        try:
            return self.get_risk_summary() or {}
        except Exception as e:
            logger.debug("Risk summary unavailable: %s", e)
            return {}

    def _engines(self) -> Dict[str, Any]:
        """Arbitrage + lifecycle context, which lives outside the trade ledger."""
        engines: Dict[str, Any] = {}
        if self.get_arb_stats:
            try:
                engines["arb"] = self.get_arb_stats() or {}
            except Exception as e:
                logger.debug("Arb stats unavailable: %s", e)
        if self.get_lifecycle_stats:
            try:
                engines["lifecycle"] = self.get_lifecycle_stats() or {}
            except Exception as e:
                logger.debug("Lifecycle stats unavailable: %s", e)
        return engines

    def _opening_balance(self, window_start_iso: str) -> Optional[float]:
        try:
            return self.trade_logger.get_balance_at(window_start_iso)
        except Exception as e:
            logger.debug("Opening balance unavailable: %s", e)
            return None

    def _collect_window(self, window_start_iso: str) -> Dict[str, Any]:
        """Everything the formatter needs for one reporting window."""
        trades: List[Dict[str, Any]] = self.trade_logger.get_closed_trades_since(window_start_iso)
        transactions: List[Dict[str, Any]] = self.trade_logger.get_transactions_since(window_start_iso)
        return {
            "trades": trades,
            "transactions": transactions,
            "window": self.trade_logger.get_performance(window_start_iso),
            "session": self.trade_logger.get_performance(),
        }

    # ── Reports ────────────────────────────────────────────────────────

    def _send_trade_digest(self, window_start_epoch: float, window_end_epoch: float) -> bool:
        """
        Send the periodic trade digest. Returns False (sends nothing) when the
        window was quiet.
        """
        window_start_iso = self._since(window_start_epoch)
        start_dt, end_dt = self._window_bounds(window_start_epoch, window_end_epoch)

        data = self._collect_window(window_start_iso)
        message = build_digest(
            data["trades"],
            data["window"],
            data["session"],
            window_start=start_dt,
            window_end=end_dt,
            opening_balance=self._opening_balance(window_start_iso),
            transactions=data["transactions"],
            risk=self._risk(),
            engines=self._engines(),
            uptime_hours=self._uptime_hours(),
            mode=self.mode,
            window_label=self._window_label(),
            max_trades=self.max_trades_per_report,
        )

        if message is None:
            logger.info(
                "Trade digest skipped: no trades closed in the last %ds",
                self.status_interval,
            )
            return False

        self.notifier.send_trade_digest(message)
        self._last_digest_count += 1
        logger.info(
            "Trade digest sent: %d closed, net $%.4f",
            data["window"]["total_trades"], data["window"]["net_pnl"],
        )
        return True

    def _send_hourly_report(self, window_start_epoch: float, window_end_epoch: float) -> bool:
        """Send the hourly rollup. Returns False when no trade closed."""
        window_start_iso = self._since(window_start_epoch)
        start_dt, end_dt = self._window_bounds(window_start_epoch, window_end_epoch)

        data = self._collect_window(window_start_iso)
        message = format_hourly_report(
            data["trades"],
            data["window"],
            data["session"],
            window_start=start_dt,
            window_end=end_dt,
            opening_balance=self._opening_balance(window_start_iso),
            risk=self._risk(),
            engines=self._engines(),
            uptime_hours=self._uptime_hours(),
            mode=self.mode,
        )

        if message is None:
            logger.info("Hourly rollup skipped: no trades closed in the last hour")
            return False

        self.notifier.send_trade_digest(message)
        logger.info(
            "Hourly report sent: %d closed, net $%.4f",
            data["window"]["total_trades"], data["window"]["net_pnl"],
        )
        return True

    def _window_label(self) -> str:
        minutes = max(1, round(self.status_interval / 60))
        return f"{minutes}-MIN"

    # ── Manual triggers ────────────────────────────────────────────────

    def force_status_ping(self) -> str:
        """
        Immediately send a status digest (for manual /status commands).

        Unlike the periodic digest this always answers: an operator who asked
        a question and got silence would read it as a dead bot.
        """
        window_start_iso = self._since(time.time() - 3600)
        try:
            window = self.trade_logger.get_performance(window_start_iso)
            session = self.trade_logger.get_performance()
            message = format_forced_digest(
                window,
                session,
                risk=self._risk(),
                uptime_hours=self._uptime_hours(),
                mode=self.mode,
                window_label="STATUS",
            )
            self.notifier.send_trade_digest(message)
            logger.info("Manual status digest sent: %d closed in last hour", window["total_trades"])
            return message
        except Exception as e:
            logger.error("Manual status digest failed: %s", e)
            return f"Status unavailable: {e}"

    def force_hourly_report(self) -> str:
        """Immediately send a rollup of the last hour (for /report)."""
        try:
            sent = self._send_hourly_report(time.time() - 3600, time.time())
            if not sent:
                message = (
                    "😴 No trades closed in the last hour — nothing to report.\n"
                    + format_forced_digest(
                        self.trade_logger.get_performance(time.time() - 3600),
                        self.trade_logger.get_performance(),
                        risk=self._risk(),
                        uptime_hours=self._uptime_hours(),
                        mode=self.mode,
                        window_label="REPORT",
                    )
                )
                self.notifier.send_trade_digest(message)
                return message
            return "Hourly report sent"
        except Exception as e:
            logger.error("Manual report failed: %s", e)
            return f"Report unavailable: {e}"
