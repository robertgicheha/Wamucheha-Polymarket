"""
Tests for data.trade_logger — in particular the log_exit metadata-merge fix
(it used to accept a `metadata` argument and silently discard it instead of
writing it to the row, which would have broken shadow-mode Brier tracking).
"""
import tempfile
from pathlib import Path

import pytest

from data.trade_logger import TradeLogger


@pytest.fixture
def logger(tmp_path):
    return TradeLogger(db_path=str(tmp_path / "test_trades.db"))


def test_log_entry_then_exit_merges_metadata(logger):
    trade_id = logger.log_entry(
        condition_id="0xabc",
        asset="btc",
        side="YES",
        price=0.62,
        size_usd=50.0,
        strategy="kelly",
        bankroll_after=50.0,
        metadata={"price_to_beat": 100000.0, "model_prob": 0.71},
    )

    logger.log_exit(
        trade_id=trade_id,
        exit_price=0.78,
        exit_reason="resolution",
        pnl_usd=12.80,
        fees_usd=0.92,
        bankroll_after=62.80,
        metadata={"won": True, "final_price": 101200.0},
    )

    rows = logger.get_recent_trades(limit=1)
    assert len(rows) == 1
    row = rows[0]
    assert row["pnl_usd"] == pytest.approx(12.80)
    assert row["won"] == 1

    import json
    meta = json.loads(row["metadata_json"])
    # Both entry-time and exit-time fields must survive.
    assert meta["price_to_beat"] == 100000.0
    assert meta["model_prob"] == 0.71
    assert meta["won"] is True
    assert meta["final_price"] == 101200.0


def test_log_exit_without_metadata_does_not_wipe_entry_metadata(logger):
    trade_id = logger.log_entry(
        condition_id="0xdef", asset="eth", side="NO", price=0.4,
        size_usd=10.0, bankroll_after=90.0,
        metadata={"model_prob": 0.2},
    )
    logger.log_exit(
        trade_id=trade_id, exit_price=0.1, exit_reason="resolution",
        pnl_usd=5.0, bankroll_after=95.0,
    )
    import json
    row = logger.get_recent_trades(limit=1)[0]
    meta = json.loads(row["metadata_json"])
    assert meta["model_prob"] == 0.2
