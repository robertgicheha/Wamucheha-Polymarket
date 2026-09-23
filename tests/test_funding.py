"""Tests for execution/funding.py guard rails (no network, no withdrawals)."""
import time

import pytest

import execution.funding as funding
from config.settings import settings
from execution.funding import FundingManager

DEST = "0x" + "ab" * 20
WALLET = "0x" + "cd" * 20


class FakeExchange:
    def __init__(self, balance=500.0):
        self.balance = balance
        self.sent = []

    def get_usdc_balance(self):
        return self.balance

    def withdraw_usdc_polygon(self, amount, to):
        self.sent.append((amount, to))
        return f"wd{len(self.sent)}"


@pytest.fixture
def mgr(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "trading_mode", "live")
    monkeypatch.setattr(settings, "funding_deposit_address", DEST)
    monkeypatch.setattr(settings, "funding_max_withdrawal_usd", 100.0)
    monkeypatch.setattr(settings, "funding_max_daily_usd", 150.0)
    monkeypatch.setattr(funding, "get_bridge_evm_address", lambda wallet: DEST)
    ex = FakeExchange()
    return FundingManager(okx=ex, ledger_path=str(tmp_path / "ledger.json")), ex


def test_fund_sends_only_to_verified_address(mgr):
    manager, ex = mgr
    assert manager.fund("okx", 50, WALLET) == "wd1"
    assert ex.sent == [(50, DEST)]


def test_refuses_when_destination_is_not_our_bridge_address(mgr, monkeypatch):
    manager, ex = mgr
    monkeypatch.setattr(funding, "get_bridge_evm_address", lambda wallet: "0x" + "99" * 20)
    with pytest.raises(RuntimeError, match="not the Polymarket bridge address"):
        manager.fund("okx", 50, WALLET)
    assert ex.sent == []


def test_per_withdrawal_and_daily_caps_and_cooldown(mgr):
    manager, ex = mgr
    with pytest.raises(ValueError, match="FUNDING_MAX_WITHDRAWAL_USD"):
        manager.fund("okx", 101, WALLET)
    with pytest.raises(ValueError, match="under \\$2"):
        manager.fund("okx", 1, WALLET)
    manager.fund("okx", 100, WALLET)
    with pytest.raises(ValueError, match="cooldown"):
        manager.fund("okx", 10, WALLET)
    manager._ledger[0]["ts"] = time.time() - 3600  # past cooldown, still inside 24h
    with pytest.raises(ValueError, match="24h funding cap"):
        manager.fund("okx", 60, WALLET)


def test_daily_cap_survives_restart(mgr):
    manager, ex = mgr
    manager.fund("okx", 100, WALLET)
    reloaded = FundingManager(okx=ex, ledger_path=manager.ledger_path)
    assert reloaded.withdrawn_last_24h() == pytest.approx(100)


def test_refuses_outside_live_mode(mgr, monkeypatch):
    manager, ex = mgr
    monkeypatch.setattr(settings, "trading_mode", "paper")
    with pytest.raises(RuntimeError, match="live"):
        manager.fund("okx", 50, WALLET)


def test_auto_fund_only_when_enabled_and_low(mgr, monkeypatch):
    manager, ex = mgr
    monkeypatch.setattr(settings, "funding_min_balance_usd", 20.0)
    monkeypatch.setattr(settings, "funding_topup_usd", 50.0)
    monkeypatch.setattr(settings, "auto_funding_enabled", False)
    assert manager.maybe_auto_fund(5.0, WALLET) is None
    monkeypatch.setattr(settings, "auto_funding_enabled", True)
    assert manager.maybe_auto_fund(30.0, WALLET) is None
    assert manager.maybe_auto_fund(5.0, WALLET) == "wd1"
