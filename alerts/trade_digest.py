"""
trade_digest.py — Renders trade activity as a readable, intentional report.

Design rules, all of them learned from the status ping this replaces:

  * Plain text, no markdown. The notifier posts without a `parse_mode`, so
    `**bold**` arrives as literal asterisks on Telegram. Structure comes from
    emoji, arrows and indentation instead, which render identically on
    Telegram, Discord and in the log file.
  * A report is about *what happened*, not about the bot being alive. Silence
    is a valid state: `build_digest` returns None when nothing closed, so
    nothing is sent at all rather than a filler "no trades" ping.
  * Every number is one a human would ask about — what was traded, when, at
    what price, how much went out, how much came back, what the balance did,
    and how the window and the session are trending.

Pure functions only: dicts in, string out, no I/O. That keeps the formatting
testable without a database or a network.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

# ── Presentation constants ─────────────────────────────────────────────

DIVIDER = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
INDENT = "   "

# Transaction types that belong to the trade flow. They are already fully
# represented by the per-trade breakdown, so surfacing them again as "money
# movements" would double-count every settlement.
TRADE_FLOW_TX_TYPES = frozenset({"trade_cost", "pnl_credit", "pnl_debit", "fee", "stake"})

MOVEMENT_ICONS = {
    "deposit": ("⬆️", "Deposit"),
    "withdrawal": ("⬇️", "Withdrawal"),
    "compound_in": ("♻️", "Compounded"),
    "redeem": ("💵", "Redeemed"),
    "reward": ("🎁", "Reward"),
}

EXIT_REASON_ICONS = {
    "resolution": "🎲 resolved",
    "early_exit": "🏃 early exit",
    "take_profit": "🎯 take profit",
    "stop_loss": "🛑 stop loss",
    "timeout": "⏱️ timed out",
    "manual": "✋ manual",
}

ASSET_EMOJI = {
    "btc": "₿",
    "eth": "Ξ",
    "sol": "S",
    "xrp": "X",
    "gold": "🥇",
}


# ── Scalar formatting ──────────────────────────────────────────────────


def usd(value: float, places: int = 2) -> str:
    """`$12.29` — magnitude only, no sign."""
    return f"${abs(value):,.{places}f}"


def signed_usd(value: float, places: int = 2) -> str:
    """`+$11.65` / `-$8.23` — sign is always explicit."""
    return f"{'+' if value >= 0 else '-'}${abs(value):,.{places}f}"


def signed_pct(value: float, places: int = 1) -> str:
    return f"{'+' if value >= 0 else '-'}{abs(value):.{places}f}%"


def arrow(value: float) -> str:
    """`▲` / `▼` / `▬` — direction without colour dependence."""
    if value > 0:
        return "▲"
    if value < 0:
        return "▼"
    return "▬"


def trend(value: float) -> str:
    """A signed value with its direction arrow, for balances and PnL."""
    return f"{arrow(value)} {signed_usd(value)}"


def _iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _clock(value: Optional[str]) -> str:
    parsed = _iso(value)
    return parsed.astimezone(timezone.utc).strftime("%H:%M:%S") if parsed else "--:--:--"


def _duration(start: Optional[str], end: Optional[str]) -> str:
    a, b = _iso(start), _iso(end)
    if not a or not b:
        return ""
    seconds = max(0.0, (b - a).total_seconds())
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s" if secs else f"{minutes}m"


def _asset_label(asset: str) -> str:
    """`₿ BTC` — a glyph plus the ticker, so it reads at a glance."""
    key = (asset or "").lower()
    name = key.upper() or "?"
    icon = ASSET_EMOJI.get(key)
    return f"{icon} {name}" if icon else name


def _side_label(side: str) -> str:
    """`▲ UP` / `▼ DOWN` — the arrow is the up/down market's own direction."""
    if (side or "").upper() == "YES":
        return "▲ UP"
    if (side or "").upper() == "NO":
        return "▼ DOWN"
    return (side or "?").upper()


def _truncate(text: str, limit: int = 56) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _trade_metadata(trade: Dict[str, Any]) -> Dict[str, Any]:
    import json

    raw = trade.get("metadata_json")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _shares(trade: Dict[str, Any]) -> float:
    """
    Share count for the ticket line.

    The executor records it in metadata rather than as a column, so read it
    from there and fall back to deriving it from stake ÷ entry price.
    """
    meta = _trade_metadata(trade)
    raw = meta.get("shares", trade.get("shares"))
    try:
        if raw is not None and float(raw) > 0:
            return float(raw)
    except (TypeError, ValueError):
        pass
    price = float(trade.get("entry_price") or 0.0)
    stake = float(trade.get("size_usd") or 0.0)
    return stake / price if price > 0 else 0.0


# ── Per-trade block ────────────────────────────────────────────────────


def format_trade(
    trade: Dict[str, Any],
    balance_before: Optional[float] = None,
) -> List[str]:
    """
    Render one closed trade as a self-contained block.

    `balance_before` is threaded in by the caller so the before → after pair
    is continuous across the block of trades in a window.
    """
    pnl = float(trade.get("pnl_usd") or 0.0)
    stake = float(trade.get("size_usd") or 0.0)
    fees = float(trade.get("fees_usd") or 0.0)
    entry_price = float(trade.get("entry_price") or 0.0)
    exit_price = float(trade.get("exit_price") or 0.0)
    won = pnl > 0
    meta = _trade_metadata(trade)

    # Proceeds credited back to the balance: the stake plus whatever PnL the
    # trade made (or lost) on the way out.
    returned = stake + pnl
    balance_after = trade.get("bankroll_after") or 0.0
    roi = (pnl / stake * 100) if stake > 0 else 0.0

    icon = "🟢" if won else "🔴"
    verdict = "WON" if won else "LOST"
    reason = EXIT_REASON_ICONS.get(
        (trade.get("exit_reason") or "").lower(),
        f"📌 {trade.get('exit_reason') or 'closed'}",
    )
    side = _side_label(trade.get("entry_side", ""))
    asset = _asset_label(trade.get("asset", ""))

    lines = [
        f"{icon} {asset} {side} · {verdict} · {reason}",
        f"{INDENT}⏱️  {_clock(trade.get('entry_time'))} → {_clock(trade.get('exit_time'))}"
        f"  ({_duration(trade.get('entry_time'), trade.get('exit_time'))})",
    ]

    ticket = f"{INDENT}🎫  {_shares(trade):.2f} sh"
    lines.append(
        f"{ticket}  entry {entry_price:.3f} → exit {exit_price:.3f}"
        f"  ({(exit_price - entry_price) * 100:+.1f}¢)"
    )
    lines.append(
        f"{INDENT}💵  Staked {usd(stake)} · Returned {usd(returned)}"
        f" · Fees {usd(fees, 4)}"
    )

    pnl_icon = "📈" if won else "📉"
    pnl_line = f"{INDENT}{pnl_icon}  PnL {signed_usd(pnl)} ({signed_pct(roi)})"
    if balance_before is not None and balance_after:
        pnl_line += f"  ·  🏦 {usd(balance_before)} → {usd(balance_after)} {arrow(pnl)}"
    lines.append(pnl_line)

    edge = meta.get("signal_edge")
    strike = meta.get("price_to_beat")
    context = []
    if isinstance(strike, (int, float)) and strike > 0:
        context.append(f"strike {strike:,.2f}")
    if isinstance(edge, (int, float)):
        context.append(f"edge {edge:+.3f}")
    if context:
        lines.append(f"{INDENT}🧠  {' · '.join(context)}")

    question = trade.get("market_question") or trade.get("condition_id") or ""
    if question:
        lines.append(f"{INDENT}📰  {_truncate(question)}")

    return lines


# ── Section builders ──────────────────────────────────────────────────


def _window_section(perf: Dict[str, Any]) -> List[str]:
    """What the reporting window itself did."""
    total = perf.get("total_trades", 0)
    wins = perf.get("wins", 0)
    losses = perf.get("losses", 0)
    net = perf.get("net_pnl", 0.0)
    staked = perf.get("total_staked", 0.0)
    returned = perf.get("total_returned", 0.0)
    fees = perf.get("total_fees", 0.0)
    efficiency = perf.get("efficiency_pct", 0.0)

    head_icon = "📈" if net > 0 else "📉" if net < 0 else "➖"
    lines = [
        "🧾 THIS WINDOW",
        f"{INDENT}🎯  {total} closed · {perf.get('win_rate_pct', 0):.0f}% accuracy"
        f"  ({wins}🟢 {losses}🔴)",
        f"{INDENT}{head_icon}  Net {signed_usd(net)} · Fees {usd(fees, 4)}"
        f" · After fees {signed_usd(perf.get('net_after_fees', 0.0))}",
        f"{INDENT}📊  Efficiency {signed_pct(efficiency)}  (net ÷ {usd(staked)} staked)",
        f"{INDENT}💼  Returned {usd(returned)} on {usd(staked)} staked",
    ]

    opening = perf.get("opening_balance") or 0.0
    closing = perf.get("end_balance") or 0.0
    if closing:
        lines.append(
            f"{INDENT}🏦  Balance {usd(opening)} → {usd(closing)}"
            f"  ({arrow(perf.get('balance_delta', 0.0))} {signed_usd(perf.get('balance_delta', 0.0))},"
            f" {signed_pct(perf.get('return_pct', 0.0))})"
        )

    factor = perf.get("profit_factor")
    if factor is not None:
        lines.append(f"{INDENT}⚖️  Profit factor {factor:.2f}")
    if perf.get("largest_win") or perf.get("largest_loss"):
        lines.append(
            f"{INDENT}🏅  Best {signed_usd(perf.get('largest_win', 0.0))}"
            f" · Worst {signed_usd(perf.get('largest_loss', 0.0))}"
        )
    return lines


def _session_section(perf: Dict[str, Any], risk: Optional[Dict[str, Any]], uptime_hours: float) -> List[str]:
    """Lifetime context so each window can be judged against the trend."""
    net = perf.get("net_pnl", 0.0)
    total = perf.get("total_trades", 0)
    lines = [
        f"📈 SESSION · up {uptime_hours:.1f}h",
        f"{INDENT}🎯  {total} closed · {perf.get('win_rate_pct', 0):.0f}% accuracy"
        f"  ({perf.get('wins', 0)}🟢 {perf.get('losses', 0)}🔴)"
        f" · Net {signed_usd(net)}",
        f"{INDENT}📊  Efficiency {signed_pct(perf.get('efficiency_pct', 0.0))}"
        f"  · Fees {usd(perf.get('total_fees', 0.0), 4)}",
    ]

    opening = perf.get("opening_balance") or 0.0
    closing = perf.get("end_balance") or 0.0
    if closing:
        lines.append(
            f"{INDENT}🏦  Bankroll {usd(opening)} → {usd(closing)}"
            f"  ({arrow(perf.get('balance_delta', 0.0))} {signed_usd(perf.get('balance_delta', 0.0))},"
            f" {signed_pct(perf.get('return_pct', 0.0))})"
        )

    if risk:
        extras = []
        drawdown = risk.get("drawdown_pct")
        if isinstance(drawdown, (int, float)) and drawdown:
            extras.append(f"📉 Drawdown {drawdown:.1f}%")
        peak = risk.get("peak_bankroll")
        if isinstance(peak, (int, float)) and peak:
            extras.append(f"⛰️  Peak {usd(peak)}")
        # RiskManager.get_compounding_summary() names this `open_positions_count`;
        # older callers used `open_positions`. Accept either so the line is
        # never silently zero.
        open_positions = risk.get("open_positions", risk.get("open_positions_count"))
        if isinstance(open_positions, (int, float)):
            extras.append(f"📂 Open {int(open_positions)}")
        if risk.get("halted"):
            extras.append("🛑 HALTED")
        if extras:
            lines.append(f"{INDENT}{' · '.join(extras)}")

    return lines


def _movements_section(transactions: Sequence[Dict[str, Any]]) -> List[str]:
    """
    Non-trade money movements: deposits, withdrawals, compounding, redeems.

    Trade-flow ledger rows are filtered out because the per-trade blocks above
    already show that money in full.
    """
    movements = [t for t in transactions if t.get("tx_type") not in TRADE_FLOW_TX_TYPES]
    if not movements:
        return []
    lines = ["🏦 MONEY MOVED"]
    for tx in movements:
        icon, label = MOVEMENT_ICONS.get(
            tx.get("tx_type", ""), ("💠", (tx.get("tx_type") or "movement").title())
        )
        amount = float(tx.get("amount_usd") or 0.0)
        balance = float(tx.get("balance_after") or 0.0)
        line = f"{INDENT}{icon}  {label} {signed_usd(amount)} → balance {usd(balance)}"
        description = tx.get("description")
        if description:
            line += f"  ·  {_truncate(description, 40)}"
        lines.append(line)
    return lines


def _asset_breakdown(perf: Dict[str, Any]) -> List[str]:
    """Per-asset attribution — which market is actually carrying the PnL."""
    by_asset = perf.get("by_asset") or {}
    if not by_asset:
        return []
    lines = ["🗺  BY ASSET"]
    for asset, stats in sorted(by_asset.items(), key=lambda kv: kv[1]["pnl"]):
        trades = stats.get("trades", 0)
        accuracy = (stats.get("wins", 0) / trades * 100) if trades else 0.0
        lines.append(
            f"{INDENT}{_asset_label(asset)}  {trades} closed · {accuracy:.0f}% accuracy"
            f" · Net {signed_usd(stats.get('pnl', 0.0))}"
        )
    return lines


def _engines_section(engines: Optional[Dict[str, Any]]) -> List[str]:
    """
    Cross-engine PnL that never reaches the trade ledger (arbitrage) plus
    live exposure, rendered only when there is something to say. A line of
    zeros is exactly the filler this report is trying to avoid, so both
    engines stay silent until they hold a position or have moved money.
    """
    if not engines:
        return []

    lines: List[str] = []
    arb = engines.get("arb") or {}
    arb_open = arb.get("open_positions") or 0
    arb_pnl = arb.get("total_pnl") or 0.0
    if arb_open or arb_pnl:
        lines.append("⚡ ENGINES")
        lines.append(
            f"{INDENT}🔀  Arbitrage PnL {signed_usd(arb_pnl)}"
            f" · {arb.get('opportunities_taken', 0)} taken"
            f" / {arb.get('total_opportunities', 0)} seen"
            f" · {arb_open} open"
        )

    lifecycle = engines.get("lifecycle") or {}
    active = lifecycle.get("active") or 0
    if active:
        lines.append(
            f"{INDENT}⏱️   5-min markets: {active} live"
            f" · {lifecycle.get('traded', 0)} traded"
            f" · {lifecycle.get('win_rate', 0):.0f}% accuracy"
        )
    return lines


# ── Public entry point ────────────────────────────────────────────────


def build_digest(
    trades: Sequence[Dict[str, Any]],
    window: Dict[str, Any],
    session: Dict[str, Any],
    *,
    window_start: Optional[datetime] = None,
    window_end: Optional[datetime] = None,
    opening_balance: Optional[float] = None,
    transactions: Sequence[Dict[str, Any]] = (),
    risk: Optional[Dict[str, Any]] = None,
    engines: Optional[Dict[str, Any]] = None,
    uptime_hours: float = 0.0,
    mode: str = "PAPER",
    window_label: str = "5-MIN",
    max_trades: int = 6,
) -> Optional[str]:
    """
    Build the periodic trade digest, or None when there is nothing to say.

    Returning None is the whole point of the skip rule: a window in which
    nothing closed and no money moved produces no message at all rather than
    a hollow ping. `trades` may be empty as long as money actually moved (a
    deposit, a withdrawal, a redemption) — that is still worth reporting.
    """
    movements = [t for t in transactions if t.get("tx_type") not in TRADE_FLOW_TX_TYPES]
    engine_lines = _engines_section(engines)
    if not trades and not movements and not engine_lines:
        return None

    now = window_end or datetime.now(timezone.utc)
    start = window_start or now
    mode_icon = "🟢" if str(mode).upper() == "LIVE" else "🧪"
    lines = [
        f"{mode_icon} {window_label} TRADE DIGEST · {mode} · "
        f"{_stamp(start)} → {_stamp(now)} UTC",
        DIVIDER,
    ]

    for trade in list(trades)[-max_trades:]:
        balance_before = _balance_before(trade, trades, opening_balance)
        lines.extend(format_trade(trade, balance_before))

    hidden = len(trades) - max_trades
    if hidden > 0:
        lines.append(f"{INDENT}… and {hidden} more closed in this window")

    if not trades:
        lines.append(f"{INDENT}😴  No trades closed in this window.")

    movement_lines = _movements_section(transactions)
    if movement_lines:
        lines.append("")
        lines.extend(movement_lines)

    if engine_lines:
        lines.append("")
        lines.extend(engine_lines)

    lines.append("")
    lines.append(DIVIDER)
    lines.extend(_window_section(window))

    if session and session.get("total_trades", 0) > (window.get("total_trades", 0) or 0):
        lines.append("")
        lines.extend(_session_section(session, risk, uptime_hours))

    breakdown = _asset_breakdown(window)
    if breakdown:
        lines.append("")
        lines.extend(breakdown)

    return "\n".join(lines)


def _stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%H:%M")


def _balance_before(
    trade: Dict[str, Any],
    trades: Sequence[Dict[str, Any]],
    opening_balance: Optional[float],
) -> Optional[float]:
    """
    Balance immediately before this trade's stake left the account.

    The ledger records `balance_at_entry` (post-stake) per trade, so adding
    the stake back recovers the true opening balance for that trade without
    depending on which trades happen to share the report window. Older rows
    written before that column existed fall back to the window opening, then
    to the previous trade's closing balance.
    """
    entry_balance = trade.get("balance_at_entry") or 0.0
    stake = trade.get("size_usd") or 0.0
    if entry_balance:
        return entry_balance + stake

    if opening_balance:
        return opening_balance

    index = None
    for i, candidate in enumerate(trades):
        if candidate.get("trade_id") == trade.get("trade_id"):
            index = i
            break
    if index is not None and index > 0:
        previous = trades[index - 1].get("bankroll_after") or 0.0
        if previous:
            return previous

    return None


def format_hourly_report(
    trades: Sequence[Dict[str, Any]],
    window: Dict[str, Any],
    session: Dict[str, Any],
    *,
    window_start: Optional[datetime] = None,
    window_end: Optional[datetime] = None,
    opening_balance: Optional[float] = None,
    risk: Optional[Dict[str, Any]] = None,
    engines: Optional[Dict[str, Any]] = None,
    uptime_hours: float = 0.0,
    mode: str = "PAPER",
) -> Optional[str]:
    """
    The hourly rollup: every trade that closed in the hour, then the hour's
    attribution, then the session trend. Returns None when nothing closed.
    """
    if not trades:
        return None

    now = window_end or datetime.now(timezone.utc)
    start = window_start or now
    mode_icon = "🟢" if str(mode).upper() == "LIVE" else "🧪"
    stamp = now.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"{mode_icon} HOURLY ROLLUP · {mode} · {stamp}",
        f"🕐 Window {_stamp(start)} → {_stamp(now)} UTC · up {uptime_hours:.1f}h",
        DIVIDER,
    ]

    for trade in trades:
        balance_before = _balance_before(trade, trades, opening_balance)
        lines.extend(format_trade(trade, balance_before))

    lines.append("")
    lines.append(DIVIDER)
    lines.extend(_window_section(window))
    lines.append("")
    lines.extend(_session_section(session, risk, uptime_hours))

    engine_lines = _engines_section(engines)
    if engine_lines:
        lines.append("")
        lines.extend(engine_lines)

    breakdown = _asset_breakdown(window)
    if breakdown:
        lines.append("")
        lines.extend(breakdown)

    return "\n".join(lines)


def format_forced_digest(
    window: Dict[str, Any],
    session: Dict[str, Any],
    *,
    risk: Optional[Dict[str, Any]] = None,
    uptime_hours: float = 0.0,
    mode: str = "PAPER",
    window_label: str = "ON-DEMAND",
) -> str:
    """
    Digest for an explicit operator request.

    Unlike the periodic digest this always renders — asking for a status and
    getting nothing back would read as a broken bot. With no trades in the
    last hour it reports the session state and says plainly that the window
    was quiet.
    """
    mode_icon = "🟢" if str(mode).upper() == "LIVE" else "🧪"
    now = datetime.now(timezone.utc)
    lines = [
        f"{mode_icon} {window_label} TRADE DIGEST · {mode} · {now.strftime('%H:%M')} UTC",
        DIVIDER,
    ]

    if window.get("total_trades", 0):
        lines.extend(_window_section(window))
        breakdown = _asset_breakdown(window)
        if breakdown:
            lines.append("")
            lines.extend(breakdown)
    else:
        lines.append(f"{INDENT}😴  No trades closed in the last hour — nothing to report.")

    lines.append("")
    lines.extend(_session_section(session, risk, uptime_hours))
    return "\n".join(lines)
