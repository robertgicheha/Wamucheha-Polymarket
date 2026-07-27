"""
scheduler.py — Periodic reporting and status pings.

Runs two jobs:
  1. Every 5 minutes: compact status ping to Telegram + Discord
  2. Every hour: full rollup report (profit, loss, win%, fees, etc.)

Both jobs read from TradeLogger and push via the alert notifier.
"""

import logging
import time
import threading
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class BotScheduler:
    """
    Lightweight scheduler that runs periodic jobs in a background thread.

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
    ):
        self.trade_logger = trade_logger
        self.notifier = notifier
        self.status_interval = status_interval_seconds
        self.report_interval = report_interval_seconds
        self.get_risk_summary = get_risk_summary
        self.get_arb_stats = get_arb_stats
        self.get_lifecycle_stats = get_lifecycle_stats

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_status_time = 0.0
        self._last_report_time = 0.0
        self._start_time = time.time()

    def start(self):
        """Start the scheduler in a background thread."""
        if self._running:
            return
        self._running = True
        self._last_status_time = time.time()
        self._last_report_time = time.time()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="bot-scheduler")
        self._thread.start()
        logger.info(
            "Scheduler started: status every %ds, report every %ds",
            self.status_interval, self.report_interval,
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

            # 5-minute status ping
            if now - self._last_status_time >= self.status_interval:
                try:
                    self._send_status_ping()
                except Exception as e:
                    logger.error("Status ping failed: %s", e)
                self._last_status_time = now

            # Hourly report
            if now - self._last_report_time >= self.report_interval:
                try:
                    self._send_hourly_report()
                except Exception as e:
                    logger.error("Hourly report failed: %s", e)
                self._last_report_time = now

            time.sleep(10)  # check every 10 seconds

    def _send_status_ping(self):
        """Send a compact 5-minute status ping."""
        report = self.trade_logger.get_hourly_report(hours=1)
        uptime_hours = (time.time() - self._start_time) / 3600

        # Build compact status message
        lines = [
            f"**5-Min Status Ping** | {datetime.now(timezone.utc).strftime('%H:%M UTC')}",
            f"Uptime: {uptime_hours:.1f}h",
        ]

        if report["total_trades"] > 0:
            lines.append(
                f"Last hour: {report['total_trades']} trades | "
                f"Win: {report['win_rate_pct']:.0f}% | "
                f"PnL: ${report['net_pnl']:+.4f}"
            )
        else:
            lines.append("Last hour: no trades")

        # Bankroll from report
        if report["current_bankroll"] > 0:
            lines.append(f"Bankroll: ${report['current_bankroll']:.2f}")

        # Risk summary if available
        if self.get_risk_summary:
            try:
                risk = self.get_risk_summary()
                if risk:
                    lines.append(f"Drawdown: {risk.get('drawdown_pct', 0):.1f}% | Open: {risk.get('open_positions', 0)}")
            except Exception:
                pass

        # Lifecycle stats if available
        if self.get_lifecycle_stats:
            try:
                lc = self.get_lifecycle_stats()
                if lc:
                    lines.append(f"5-min markets: {lc.get('active', 0)} active | PnL: ${lc.get('pnl', 0):+.4f}")
            except Exception:
                pass

        msg = "\n".join(lines)
        from alerts.notifier import Severity
        self.notifier.send(msg, Severity.INFO)
        logger.debug("Status ping sent: %d trades in last hour", report["total_trades"])

    def _send_hourly_report(self):
        """Send a comprehensive hourly rollup report."""
        report = self.trade_logger.get_hourly_report(hours=1)
        session = self.trade_logger.get_session_summary()
        uptime_hours = (time.time() - self._start_time) / 3600

        lines = [
            "=" * 40,
            f"**HOURLY REPORT** | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 40,
            "",
            "**This Hour:**",
            f"  Trades: {report['total_trades']} (W:{report['wins']} L:{report['losses']})",
            f"  Win Rate: {report['win_rate_pct']:.1f}%",
            f"  Gross Profit: ${report['gross_profit']:.4f}",
            f"  Gross Loss: ${report['gross_loss']:.4f}",
            f"  Net PnL: ${report['net_pnl']:+.4f}",
            f"  Fees: ${report['total_fees']:.4f}",
            f"  Volume: ${report['total_volume']:.2f}",
            f"  Avg Trade: ${report['avg_trade_size']:.2f}",
            f"  Best: ${report['largest_win']:+.4f} | Worst: ${report['largest_loss']:+.4f}",
        ]

        # Per-asset breakdown
        if report["trades_by_asset"]:
            lines.append("")
            lines.append("**By Asset:**")
            for asset, stats in report["trades_by_asset"].items():
                wr = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
                lines.append(
                    f"  {asset.upper()}: {stats['trades']} trades | "
                    f"Win: {wr:.0f}% | PnL: ${stats['pnl']:+.4f}"
                )

        # Session totals
        lines.extend([
            "",
            "**Session Totals:**",
            f"  Uptime: {uptime_hours:.1f}h",
            f"  Total Trades: {session.get('total_trades', 0)} "
            f"(W:{session.get('wins', 0)} L:{session.get('losses', 0)})",
            f"  Session Win Rate: {session.get('win_rate_pct', 0):.1f}%",
            f"  Session PnL: ${session.get('session_pnl', 0):+.4f}",
            f"  Session Fees: ${session.get('session_fees', 0):.4f}",
            f"  Bankroll: ${report['current_bankroll']:.2f}",
        ])

        # Arb stats if available
        if self.get_arb_stats:
            try:
                arb = self.get_arb_stats()
                if arb:
                    lines.extend([
                        "",
                        "**Arbitrage:**",
                        f"  PnL: ${arb.get('total_pnl', 0):+.4f} | "
                        f"Win: {arb.get('win_rate', 0):.0f}% | "
                        f"Open: {arb.get('open_positions', 0)}",
                    ])
            except Exception:
                pass

        # Lifecycle stats if available
        if self.get_lifecycle_stats:
            try:
                lc = self.get_lifecycle_stats()
                if lc:
                    lines.extend([
                        "",
                        "**5-Minute Lifecycle:**",
                        f"  Active Markets: {lc.get('active', 0)}",
                        f"  Total PnL: ${lc.get('pnl', 0):+.4f}",
                        f"  Win Rate: {lc.get('win_rate', 0):.1f}%",
                        f"  Markets Traded: {lc.get('traded', 0)}",
                    ])
            except Exception:
                pass

        msg = "\n".join(lines)
        from alerts.notifier import Severity
        self.notifier.send(msg, Severity.INFO)
        logger.info("Hourly report sent: %d trades, PnL $%.4f", report["total_trades"], report["net_pnl"])

    # ── Manual triggers ────────────────────────────────────────────────

    def force_status_ping(self):
        """Immediately send a status ping (for manual /status commands)."""
        try:
            self._send_status_ping()
        except Exception as e:
            logger.error("Manual status ping failed: %s", e)

    def force_hourly_report(self):
        """Immediately send a full report (for manual /report commands)."""
        try:
            self._send_hourly_report()
        except Exception as e:
            logger.error("Manual report failed: %s", e)
