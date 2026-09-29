"""
Tests for the periodic trade digest.

Covers the two behaviours that matter: that a window with no trades produces
no message at all, and that a window with trades produces a report a human
can act on (what was traded, when, the money in and out, and the balance).
"""
import time
from datetime import datetime, timedelta, timezone

import pytest

from alerts.trade_digest import (
    build_digest,
    format_forced_digest,
    format_hourly_report,
    signed_usd,
    usd,
)
from data.trade_logger import TradeLogger, _empty_performance

T0 = datetime(2026, 9, 29, 17, 24, tzinfo=timezone.utc)


def _trade(**overrides):
    """A closed trade shaped exactly like a row the ledger produces."""
    base = {
        "trade_id": "trd_1_btc",
        "condition_id": "0xabc",
        "asset": "btc",
        "entry_side": "YES",
        "entry_price": 0.512,
        "entry_time": T0.isoformat(),
        "exit_price": 1.0,
        "exit_time": (T0 + timedelta(minutes=3)).isoformat(),
        "exit_reason": "resolution",
        "size_usd": 12.29,
        "pnl_usd": 11.65,
        "fees_usd": 0.06,
        "won": 1,
        "balance_at_entry": 84.45,   # 96.74 - 12.29, i.e. after the stake left
        "bankroll_after": 108.39,
        "market_question": "Bitcoin Up or Down - September 29, 5PM ET",
        "metadata_json": '{"shares": 24.0, "price_to_beat": 101200.5, "signal_edge": 0.031}',
    }
    base.update(overrides)
    return base


# ── Scalar formatting ──────────────────────────────────────────────────


def test_signed_usd_always_shows_sign():
    assert signed_usd(11.654) == "+$11.65"
    assert signed_usd(-8.231) == "-$8.23"
    assert signed_usd(0) == "+$0.00"


def test_usd_drops_sign():
    assert usd(96.74) == "$96.74"
    assert usd(-8.231) == "$8.23"


# ── The skip rule ──────────────────────────────────────────────────────


def test_no_trades_and_no_money_means_no_message():
    assert build_digest([], _empty_performance(), _empty_performance()) is None


def test_trade_flow_ledger_rows_alone_do_not_wake_the_bot():
    """Entry/exit/PnL/fee rows belong to a trade; without one there is nothing."""
    transactions = [
        {"tx_type": "trade_cost", "amount_usd": -12.0, "balance_after": 88.0},
        {"tx_type": "pnl_credit", "amount_usd": 12.0, "balance_after": 100.0},
        {"tx_type": "fee", "amount_usd": -0.1, "balance_after": 99.9},
    ]
    assert build_digest([], _empty_performance(), _empty_performance(),
                        transactions=transactions) is None


def test_money_movement_alone_does_wake_the_bot():
    transactions = [
        {"tx_type": "withdrawal", "amount_usd": -20.0, "balance_after": 80.0,
         "description": "manual payout"},
    ]
    message = build_digest([], _empty_performance(), _empty_performance(),
                           transactions=transactions)
    assert message is not None
    assert "MONEY MOVED" in message
    assert "Withdrawal" in message
    assert "-$20.00" in message


# ── Content of a real digest ───────────────────────────────────────────


def test_digest_reports_what_when_and_the_money():
    trade = _trade()
    window = {
        **_empty_performance(),
        "total_trades": 1, "wins": 1, "losses": 0, "win_rate_pct": 100.0,
        "net_pnl": 11.65, "total_fees": 0.06, "net_after_fees": 11.59,
        "total_staked": 12.29, "total_returned": 23.94, "efficiency_pct": 94.8,
        "opening_balance": 96.74, "end_balance": 108.39,
        "balance_delta": 11.65, "return_pct": 12.0,
    }
    message = build_digest([trade], window, _empty_performance(),
                           window_start=T0, window_end=T0 + timedelta(minutes=5),
                           opening_balance=96.74)

    # What was traded, and which way.
    assert "BTC" in message and "UP" in message and "🟢" in message
    # When.
    assert "17:24:00" in message and "17:27:00" in message
    # Money out, money back, fees.
    assert "Staked $12.29" in message
    assert "Returned $23.94" in message
    assert "Fees $0.0600" in message
    # PnL, with direction.
    assert "PnL +$11.65 (+94.8%)" in message
    # Balance before and after.
    assert "$96.74 → $108.39" in message
    # Accuracy and efficiency.
    assert "100% accuracy" in message
    assert "Efficiency +94.8%" in message


def test_digest_marks_a_loss_with_a_down_arrow():
    trade = _trade(trade_id="trd_2_eth", asset="eth", entry_side="NO",
                   entry_price=0.455, exit_price=0.0, size_usd=8.19,
                   pnl_usd=-8.19, won=0, balance_at_entry=108.39,
                   bankroll_after=100.20, metadata_json="{}")
    window = {
        **_empty_performance(),
        "total_trades": 1, "wins": 0, "losses": 1, "win_rate_pct": 0.0,
        "net_pnl": -8.19, "total_staked": 8.19, "total_returned": 0.0,
        "efficiency_pct": -100.0, "opening_balance": 108.39, "end_balance": 100.20,
        "balance_delta": -8.19, "return_pct": -7.5,
    }
    message = build_digest([trade], window, _empty_performance(),
                           window_start=T0, window_end=T0 + timedelta(minutes=5),
                           opening_balance=108.39)

    assert "🔴" in message and "LOST" in message
    assert "Ξ ETH" in message and "▼ DOWN" in message
    assert "Returned $0.00" in message
    assert "PnL -$8.19 (-100.0%)" in message
    assert "$108.39 → $100.20" in message
    assert "0% accuracy" in message


def test_balance_before_is_the_ledger_balance_plus_the_stake():
    """
    Each trade's 'before' comes from its own balance_at_entry, so trades stay
    correct even when a report window starts mid-position.
    """
    first = _trade()
    second = _trade(trade_id="trd_2_eth", asset="eth", entry_side="NO",
                    entry_price=0.455, exit_price=0.0, size_usd=8.19,
                    pnl_usd=-8.19, won=0, balance_at_entry=100.20,
                    bankroll_after=100.20,
                    exit_time=(T0 + timedelta(minutes=5)).isoformat(),
                    metadata_json="{}")
    window = {**_empty_performance(), "total_trades": 2, "wins": 1, "losses": 1,
              "net_pnl": 3.46, "total_staked": 20.48, "efficiency_pct": 16.9}
    message = build_digest([first, second], window, _empty_performance(),
                           window_start=T0, window_end=T0 + timedelta(minutes=5),
                           opening_balance=96.74)
    assert "$96.74 → $108.39" in message
    assert "$108.39 → $100.20" in message


def test_session_block_appears_only_when_it_adds_context():
    trade = _trade()
    window = {**_empty_performance(), "total_trades": 1, "wins": 1,
              "win_rate_pct": 100.0, "net_pnl": 11.65}
    same = build_digest([trade], window, dict(window), window_start=T0,
                        window_end=T0 + timedelta(minutes=5), opening_balance=96.74)
    assert "SESSION" not in same

    session = {**_empty_performance(), "total_trades": 14, "wins": 8, "losses": 6,
               "win_rate_pct": 57.1, "net_pnl": 12.30, "efficiency_pct": 8.1,
               "opening_balance": 96.74, "end_balance": 109.04, "balance_delta": 12.30,
               "return_pct": 12.7}
    richer = build_digest([trade], window, session, window_start=T0,
                          window_end=T0 + timedelta(minutes=5), opening_balance=96.74,
                          risk={"drawdown_pct": 3.3, "open_positions_count": 1,
                                "peak_bankroll": 112.0},
                          uptime_hours=2.8)
    assert "SESSION" in richer
    assert "14 closed · 57% accuracy" in richer
    assert "Drawdown 3.3%" in richer
    assert "Open 1" in richer


def test_digest_caps_the_trade_list():
    trades = [
        _trade(trade_id=f"trd_{i}", bankroll_after=100.0 + i,
               exit_time=(T0 + timedelta(minutes=i)).isoformat())
        for i in range(10)
    ]
    window = {**_empty_performance(), "total_trades": 10}
    message = build_digest(trades, window, _empty_performance(), window_start=T0,
                           window_end=T0 + timedelta(minutes=10), max_trades=6)
    assert "… and 4 more closed in this window" in message
    assert message.count("🎫") == 6


def test_engine_line_is_omitted_until_it_means_something():
    trade = _trade()
    window = {**_empty_performance(), "total_trades": 1, "wins": 1, "net_pnl": 11.65}
    idle = build_digest([trade], window, _empty_performance(), window_start=T0,
                        window_end=T0 + timedelta(minutes=5),
                        engines={"arb": {"total_pnl": 0.0, "open_positions": 0},
                                 "lifecycle": {"active": 0}})
    assert "ENGINES" not in idle

    busy = build_digest([trade], window, _empty_performance(), window_start=T0,
                        window_end=T0 + timedelta(minutes=5),
                        engines={"arb": {"total_pnl": 1.25, "open_positions": 2,
                                         "opportunities_taken": 3,
                                         "total_opportunities": 11},
                                 "lifecycle": {"active": 4, "traded": 9, "win_rate": 61.0}})
    assert "ENGINES" in busy
    assert "Arbitrage PnL +$1.25" in busy
    assert "5-min markets: 4 live" in busy


def test_digest_contains_no_markdown_asterisks():
    """The notifier posts without parse_mode, so `**` would render literally."""
    trade = _trade()
    window = {**_empty_performance(), "total_trades": 1, "wins": 1, "net_pnl": 11.65,
              "win_rate_pct": 100.0}
    message = build_digest([trade], window, _empty_performance(), window_start=T0,
                           window_end=T0 + timedelta(minutes=5), opening_balance=96.74)
    assert "*" not in message


# ── Hourly rollup & forced digest ──────────────────────────────────────


def test_hourly_rollup_skips_a_silent_hour():
    assert format_hourly_report([], _empty_performance(), _empty_performance()) is None


def test_hourly_rollup_lists_every_trade():
    trades = [_trade(trade_id=f"trd_{i}", exit_time=(T0 + timedelta(minutes=i)).isoformat())
              for i in range(3)]
    window = {**_empty_performance(), "total_trades": 3, "wins": 2, "losses": 1,
              "net_pnl": 5.0, "win_rate_pct": 66.7}
    message = format_hourly_report(trades, window, _empty_performance(),
                                   window_start=T0, window_end=T0 + timedelta(hours=1),
                                   opening_balance=96.74, uptime_hours=2.8)
    assert "HOURLY ROLLUP" in message
    assert message.count("🎫") == 3
    assert "67% accuracy" in message


def test_forced_status_always_answers():
    message = format_forced_digest(_empty_performance(), _empty_performance(),
                                   uptime_hours=1.0)
    assert message is not None
    assert "No trades closed in the last hour" in message
    assert "SESSION" in message


# ── End-to-end against the real ledger ─────────────────────────────────


@pytest.fixture
def logger(tmp_path):
    return TradeLogger(db_path=str(tmp_path / "digest.db"))


def _record(logger, index, side, entry, exit_price, stake, pnl, pre_balance):
    """Write one entry/exit pair, tracking the balance the way the engine does."""
    trade_id = logger.log_entry(
        condition_id=f"0x{index}", asset="btc", side=side, price=entry,
        size_usd=stake, strategy="fair_value", source="lifecycle_paper",
        market_question="Bitcoin Up or Down", bankroll_after=pre_balance - stake,
        metadata={"shares": stake / entry},
    )
    logger.log_exit(trade_id=trade_id, exit_price=exit_price,
                    exit_reason="resolution", pnl_usd=pnl, fees_usd=0.05,
                    bankroll_after=pre_balance + pnl,
                    metadata={"won": pnl > 0})
    return trade_id


def test_log_exit_preserves_the_entry_balance(logger):
    """The pre-trade balance must survive the exit write, or reports lie."""
    trade_id = _record(logger, 1, "YES", 0.50, 1.0, 10.0, 10.0, 100.0)
    row = logger.get_recent_trades(limit=1)[0]
    assert row["trade_id"] == trade_id
    assert row["balance_at_entry"] == 90.0    # 100 - 10 stake
    assert row["bankroll_after"] == 110.0     # after settlement


def test_scheduler_ingest_produces_a_digest(logger):
    from data.scheduler import BotScheduler

    logger.log_transaction("deposit", 100.0, 100.0, "seed")
    _record(logger, 1, "YES", 0.50, 1.0, 10.0, 10.0, 100.0)
    _record(logger, 2, "NO", 0.40, 0.0, 10.0, -10.0, 110.0)

    scheduler = BotScheduler(trade_logger=logger, notifier=None, mode="PAPER")
    window_start = (T0 - timedelta(minutes=20)).isoformat()
    data = scheduler._collect_window(window_start)

    message = build_digest(
        data["trades"], data["window"], data["session"],
        window_start=T0, window_end=T0 + timedelta(minutes=5),
        opening_balance=scheduler._opening_balance(window_start),
        transactions=data["transactions"], mode="PAPER",
    )
    assert message is not None
    assert "THIS WINDOW" in message
    assert "2 closed · 50% accuracy" in message
    # A win then a loss of the same size nets out, and the balance says so.
    assert "Net +$0.00" in message
    assert "Balance $100.00 → $100.00" in message
    # The window's ledger opening balance is the pre-first-stake balance.
    assert data["window"]["opening_balance"] == 100.0
    assert data["window"]["balance_delta"] == 0.0


def test_scheduler_is_silent_on_a_quiet_ledger(logger):
    from data.scheduler import BotScheduler

    logger.log_transaction("deposit", 100.0, 100.0, "seed")
    _record(logger, 1, "YES", 0.50, 1.0, 10.0, 10.0, 100.0)

    sent = []
    scheduler = BotScheduler(trade_logger=logger, notifier=None)
    scheduler.notifier = type(
        "N", (), {"send_trade_digest": lambda self, m: sent.append(m)}
    )()

    # A window that opens after the last settlement saw no trades and no
    # non-trade money movement, so nothing may go out.
    future = time.time() + 60
    assert scheduler._send_trade_digest(future, future + 300) is False
    assert sent == []

    # The window containing the trade does report.
    assert scheduler._send_trade_digest(0.0, time.time() + 1) is True
    assert len(sent) == 1


def test_a_zero_interval_disables_the_job(logger):
    from data.scheduler import BotScheduler

    logger.log_transaction("deposit", 100.0, 100.0, "seed")
    _record(logger, 1, "YES", 0.50, 1.0, 10.0, 10.0, 100.0)

    sent = []
    scheduler = BotScheduler(trade_logger=logger, notifier=None,
                              status_interval_seconds=0)
    scheduler.notifier = type(
        "N", (), {"send_trade_digest": lambda self, m: sent.append(m)}
    )()

    # With the interval at 0 the loop must never fire the digest, even though
    # a trade is sitting in the ledger waiting to be reported.
    deadline = time.time() + 0.5
    scheduler.start()
    while time.time() < deadline:
        time.sleep(0.05)
    scheduler.stop()
    assert sent == []
