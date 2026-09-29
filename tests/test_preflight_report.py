"""
Tests for the preflight report renderer.

The report is the thing an operator reads at 3am when a live-money bot
refuses to start, so two properties matter most and are easy to regress:

  1. The Telegram copy must contain no ANSI escapes. main.py sends the report
     to Telegram; a stray escape code renders as literal escape garbage in the
     chat bubble, and nobody can read it at a glance.
  2. The failure ordering must put gas before approvals. Approvals are an
     on-chain transaction, so an account with no POL cannot complete them no
     matter how often the operator retries.
"""
import re

import pytest

from execution.preflight import Check, critical_failures, format_report, render_report

ANSI = re.compile(r"\033\[")


def _checks(**overrides):
    """A representative run: the exact failure mix from production."""
    defaults = [
        ("config", True, "live-trading config complete", True),
        ("secrets", False, "git not installed in this image", False),
        ("kill switch", True, "not engaged", True),
        ("polygon rpc", True, "https://rpc.example: 142ms, block age 0.6s", True),
        ("clob api", True, "CLOB V2 reachable", True),
        ("polymarket auth", True, "wallet 0xdAf6 (EOA)", True),
        ("trading approvals", False,
         "missing approvals — run: python scripts/preflight.py --setup-approvals", True),
        ("collateral", False, "pUSD balance $0.00", True),
        ("gas (POL)", False, "Low POL balance (0.0000)", True),
        ("funding address", False,
         "FUNDING_DEPOSIT_ADDRESS mismatch (expected 0xd58e9B1a)", True),
        ("market data", True, "btc-updown-5m-1790709000", True),
    ]
    return [Check(*d) for d in defaults if overrides.get(d[0], True)]


# ── colour safety ────────────────────────────────────────────────────────

def test_plain_report_has_no_ansi_escapes():
    """Telegram renders this string verbatim; escapes would be garbage."""
    out = render_report(_checks(), color=False)
    assert not ANSI.search(out), f"ANSI escapes leaked into plain output: {ANSI.findall(out)}"


def test_color_report_has_ansi_escapes():
    out = render_report(_checks(), color=True)
    assert ANSI.search(out), "colour=True produced no escapes"


def test_no_color_env_disables_colour(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    from execution.preflight import _color_enabled
    assert _color_enabled() is False


def test_force_color_env_enables_colour(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    from execution.preflight import _color_enabled
    assert _color_enabled() is True


# ── structure ─────────────────────────────────────────────────────────────

def _section_headers(out, title):
    """Lines that are section headers for `title`.

    A naive substring count is wrong: 'FUNDING' also occurs inside the
    FUNDING_DEPOSIT_ADDRESS hint. A header is short, ends with the title, and
    carries nothing after it.
    """
    return [ln for ln in out.splitlines()
            if ln.strip().endswith(title) and len(ln.strip()) <= len(title) + 4]


def test_each_section_header_appears_once():
    """Checks are grouped explicitly, so a reordering cannot duplicate a header."""
    out = render_report(_checks(), color=False)
    for title in ("CONFIGURATION", "NETWORK & WALLET", "FUNDING", "MARKETS"):
        assert len(_section_headers(out, title)) == 1, \
            f"{title} rendered {len(_section_headers(out, title))} times"


def test_section_emoji_is_stable_per_section():
    """A section must not change icon depending on which check came first."""
    out = render_report(_checks(), color=False)
    # collateral and gas both belong to NETWORK & WALLET; the header icon is
    # whatever precedes that title, and must be the same every time.
    headers = _section_headers(out, "NETWORK & WALLET")
    assert len(headers) == 1
    assert headers[0].strip().startswith("\U0001F310")


def test_all_checks_are_rendered():
    out = render_report(_checks(), color=False)
    for c in _checks():
        assert c.name in out, f"{c.name} missing from report"


# ── verdict ───────────────────────────────────────────────────────────────

def test_failures_produce_not_ready():
    out = render_report(_checks(), color=False)
    assert "NOT READY" in out
    assert "READY for live trading" not in out


def test_all_pass_produces_ready():
    checks = [Check("config", True, "live-trading config complete", True),
              Check("market data", True, "btc-updown-5m", True)]
    out = render_report(checks, color=False)
    assert "READY for live trading" in out
    assert "NOT READY" not in out


def test_non_critical_failure_warns_without_blocking():
    """A WARN must not read as 'not ready' — that is the secrets-check case."""
    checks = [Check("secrets", False, "git not installed in this image", False),
              Check("config", True, "ok", True)]
    out = render_report(checks, color=False)
    assert "READY" in out and "NOT READY" not in out
    assert critical_failures(checks) == []


def test_tally_counts_are_correct():
    out = render_report(_checks(), color=False)
    assert "6 passed" in out
    assert "4 failed" in out
    assert "1 warning" in out


# ── the ordering that actually matters ────────────────────────────────────

def test_gas_is_resolved_before_approvals():
    """Approvals are an on-chain tx; with 0 POL they cannot succeed."""
    out = render_report(_checks(), color=False)
    gas = out.index("Send POL to the trading wallet")
    approvals = out.index("--setup-approvals --yes")
    assert gas < approvals, "gas must be listed before approvals"


def test_ordered_steps_only_include_failing_checks():
    out = render_report([Check("config", True, "ok", True),
                         Check("gas (POL)", False, "0 POL", True)], color=False)
    assert "Send POL" in out
    assert "--setup-approvals" not in out
    assert "Deposit USDC" not in out


def test_gas_dependency_is_called_out():
    out = render_report(_checks(), color=False)
    assert "Approvals cost gas" in out


# ── hint extraction ───────────────────────────────────────────────────────

def test_run_command_becomes_an_arrow_hint():
    out = render_report(_checks(), color=False)
    assert "missing approvals" in out
    assert "-> python scripts/preflight.py --setup-approvals" in out


def test_expected_address_becomes_an_arrow_hint():
    out = render_report(_checks(), color=False)
    assert "expected 0xd58e9B1a" in out
    # the hint is on its own line, not appended to the message
    assert "mismatch (expected" not in out


# ── back-compat ───────────────────────────────────────────────────────────

def test_format_report_is_an_alias():
    checks = _checks()
    assert format_report(checks, color=False) == render_report(checks, color=False)


@pytest.mark.parametrize("color", [True, False])
def test_report_never_raises(color):
    render_report([], color=color)
    render_report([Check("weird name", True, "", False)], color=color)
    render_report([Check("unknown check", False, "boom", True)], color=color)
