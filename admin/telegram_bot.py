"""
telegram_bot.py — Telegram front-end for the admin API.

Setup:
  1. Message @BotFather on Telegram, `/newbot`, grab the token.
  2. Message @userinfobot (or similar) to get YOUR numeric chat_id.
  3. Fill in .env: TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_CHAT_IDS,
     ADMIN_API_URL, ADMIN_API_TOKEN.
  4. python3 telegram_bot.py   (or run via systemd, see systemd/)

Commands:
  /status            process state, halted?, last log line
  /logs [n]          last n log lines (default 30, max 200)
  /trades [n]        last n trades (default 10, max 50)
  /pnl [today|week|all]
  /balance
  /compound [on|off]  enable/disable profit compounding
  /compound_status    show compounding stats and growth rate
  /withdraw <amount>  withdraw profits to wallet
  /arb               show arbitrage opportunities and PnL
  /markets           show active 5-minute market windows
  /halt [reason]     pause trading (bot finishes any in-flight bet first)
  /resume            resume trading
  /restart           hard restart via systemd (asks for confirmation)
  /help
"""

import os

import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ALLOWED_CHAT_IDS = {
    int(x) for x in os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "").split(",") if x.strip()
}
ADMIN_API_URL = os.environ.get("ADMIN_API_URL", "http://127.0.0.1:8787")
ADMIN_API_TOKEN = os.environ["ADMIN_API_TOKEN"]

HEADERS = {"Authorization": f"Bearer {ADMIN_API_TOKEN}"}
client = httpx.AsyncClient(base_url=ADMIN_API_URL, headers=HEADERS, timeout=15)

_pending_restart_confirm: set[int] = set()


def _authorized(update: Update) -> bool:
    return update.effective_chat is not None and update.effective_chat.id in ALLOWED_CHAT_IDS


async def _reply_unauthorized(update: Update):
    await update.message.reply_text("Not authorized for this bot.")


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return await _reply_unauthorized(update)
    r = await client.get("/status")
    d = r.json()
    halted = "🔴 HALTED" if d["trading_halted"] else "🟢 trading"
    msg = (
        f"Process: {d['process']}\n"
        f"State: {halted}"
        + (f" ({d['halt_reason']})" if d.get("halt_reason") else "")
        + f"\nLast log: {d.get('last_log_line') or '—'}"
    )
    await update.message.reply_text(msg)


async def logs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return await _reply_unauthorized(update)
    n = int(context.args[0]) if context.args else 30
    r = await client.get("/logs", params={"lines": n})
    lines = r.json()["lines"]
    text = "\n".join(lines) or "(empty)"
    for chunk_start in range(0, len(text), 3800):
        await update.message.reply_text(f"```\n{text[chunk_start:chunk_start+3800]}\n```", parse_mode="Markdown")


async def trades_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return await _reply_unauthorized(update)
    n = int(context.args[0]) if context.args else 10
    r = await client.get("/trades", params={"limit": n})
    rows = r.json()["trades"]
    if not rows:
        return await update.message.reply_text("No trades found.")
    lines = []
    for t in rows:
        lines.append(
            f"{t.get('timestamp','?')} {t.get('side','?'):>4} "
            f"${t.get('stake','?')} → {t.get('result','?')} "
            f"({t.get('pnl','?')})"
        )
    await update.message.reply_text("```\n" + "\n".join(lines) + "\n```", parse_mode="Markdown")


async def pnl_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return await _reply_unauthorized(update)
    period = context.args[0] if context.args else "today"
    r = await client.get("/pnl", params={"period": period})
    d = r.json()
    await update.message.reply_text(
        f"PnL ({d['period']}): {d['total_pnl']}\n"
        f"Trades: {d['trade_count']} — {d['wins']}W/{d['losses']}L "
        f"({d['win_rate_pct']}% win rate)"
    )


async def balance_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return await _reply_unauthorized(update)
    r = await client.get("/balance")
    if r.status_code != 200:
        return await update.message.reply_text("Balance file not available.")
    await update.message.reply_text(f"```\n{r.json()}\n```", parse_mode="Markdown")


async def compound_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return await _reply_unauthorized(update)
    if not context.args:
        return await update.message.reply_text("Usage: /compound [on|off]")
    setting = context.args[0].lower()
    if setting not in ("on", "off"):
        return await update.message.reply_text("Usage: /compound [on|off]")
    enabled = setting == "on"
    r = await client.post("/compound", json={"enabled": enabled})
    if r.status_code == 200:
        state = "ENABLED" if enabled else "DISABLED"
        await update.message.reply_text(f"Profit compounding {state}")
    else:
        await update.message.reply_text(f"Failed: {r.text}")


async def compound_status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return await _reply_unauthorized(update)
    r = await client.get("/compound_status")
    if r.status_code != 200:
        return await update.message.reply_text("Compound status not available.")
    d = r.json()
    await update.message.reply_text(
        f"**Compounding:** {'ON' if d.get('enabled') else 'OFF'}\n"
        f"Principal: ${d.get('principal', 0):.2f}\n"
        f"Bankroll: ${d.get('bankroll', 0):.2f}\n"
        f"Profit: ${d.get('total_profit', 0):.2f} ({d.get('profit_pct', 0):.1f}%)\n"
        f"Withdrawn: ${d.get('total_withdrawn', 0):.2f}\n"
        f"Win Rate: {d.get('win_rate', 0):.1f}%\n"
        f"Trades: {d.get('total_trades', 0)} (W:{d.get('total_wins', 0)} L:{d.get('total_losses', 0)})",
        parse_mode="Markdown",
    )


async def withdraw_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return await _reply_unauthorized(update)
    if not context.args:
        return await update.message.reply_text("Usage: /withdraw <amount_usd>")
    try:
        amount = float(context.args[0])
    except ValueError:
        return await update.message.reply_text("Invalid amount. Usage: /withdraw 50.00")
    r = await client.post("/withdraw", json={"amount_usd": amount})
    if r.status_code == 200:
        d = r.json()
        await update.message.reply_text(
            f"Withdrawal requested: ${amount:.2f}\n"
            f"Status: {d.get('status', 'pending')}\n"
            f"Remaining profit: ${d.get('remaining_profit', 0):.2f}"
        )
    else:
        await update.message.reply_text(f"Withdrawal failed: {r.text}")


async def arb_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return await _reply_unauthorized(update)
    r = await client.get("/arb")
    if r.status_code != 200:
        return await update.message.reply_text("Arb data not available.")
    d = r.json()
    lines = [
        f"**Arbitrage Stats:**",
        f"Bankroll: ${d.get('bankroll', 0):.2f}",
        f"Open Positions: {d.get('open_positions', 0)}",
        f"Total PnL: ${d.get('total_pnl', 0):.4f}",
        f"Win Rate: {d.get('win_rate', 0):.1f}%",
        f"PM Markets: {d.get('pm_markets_tracked', 0)} | Kalshi: {d.get('kalshi_contracts_tracked', 0)}",
    ]
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def markets_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return await _reply_unauthorized(update)
    r = await client.get("/markets")
    if r.status_code != 200:
        return await update.message.reply_text("Market data not available.")
    d = r.json()
    markets = d.get("markets", [])
    if not markets:
        return await update.message.reply_text("No active market windows.")
    lines = ["**5-Minute Market Windows:**\n"]
    for m in markets[:5]:
        phase_emoji = {"discover": "D", "start": "S", "run": "R", "end": "E", "settled": "X"}.get(m.get("phase", ""), "?")
        lines.append(
            f"[{phase_emoji}] {m.get('asset', '?').upper()} | "
            f"Beat: ${m.get('price_to_beat', 0):.2f} | "
            f"YES: {m.get('current_yes_price', 0.5):.3f} | "
            f"Time: {m.get('time_remaining', 0):.0f}s | "
            f"PnL: ${m.get('pnl_usd', 0):+.4f}"
        )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def report_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return await _reply_unauthorized(update)
    period = context.args[0] if context.args else "hour"
    if period not in ("hour", "session", "daily"):
        return await update.message.reply_text("Usage: /report [hour|session|daily]")
    r = await client.get(f"/report?period={period}")
    if r.status_code != 200:
        return await update.message.reply_text("Report not available.")
    d = r.json()
    if d.get("total_trades", 0) == 0 and period == "hour":
        await update.message.reply_text("No trades in the last hour.")
        return
    lines = [
        f"**{'Session' if period == 'session' else 'Hourly' if period == 'hour' else 'Daily'} Report:**\n",
        f"Trades: {d.get('total_trades', 0)} (W:{d.get('wins', 0)} L:{d.get('losses', 0)})",
        f"Win Rate: {d.get('win_rate_pct', 0):.1f}%",
        f"Gross Profit: ${d.get('gross_profit', 0):.4f}",
        f"Gross Loss: ${d.get('gross_loss', 0):.4f}",
        f"Net PnL: ${d.get('net_pnl', 0):+.4f}",
        f"Fees: ${d.get('total_fees', 0):.4f}",
        f"Volume: ${d.get('total_volume', 0):.2f}",
        f"Best: ${d.get('largest_win', 0):+.4f} | Worst: ${d.get('largest_loss', 0):+.4f}",
        f"Bankroll: ${d.get('current_bankroll', 0):.2f}",
    ]
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def backup_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return await _reply_unauthorized(update)
    r = await client.post("/backup")
    if r.status_code == 200:
        d = r.json()
        await update.message.reply_text(f"Backup completed: {d.get('backup_path', 'unknown')}")
    else:
        await update.message.reply_text(f"Backup failed: {r.text}")


async def halt_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return await _reply_unauthorized(update)
    reason = " ".join(context.args) if context.args else "manual halt via telegram"
    actor = f"telegram:{update.effective_user.id}"
    await client.post("/halt", json={"reason": reason, "actor": actor})
    await update.message.reply_text(f"🔴 Trading halted. Reason: {reason}")


async def resume_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return await _reply_unauthorized(update)
    actor = f"telegram:{update.effective_user.id}"
    await client.post("/resume", json={"actor": actor})
    await update.message.reply_text("🟢 Trading resumed.")


async def restart_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return await _reply_unauthorized(update)
    chat_id = update.effective_chat.id
    if chat_id not in _pending_restart_confirm:
        _pending_restart_confirm.add(chat_id)
        return await update.message.reply_text(
            "This hard-restarts the bot process (not just a pause). "
            "Send /restart again within 30s to confirm."
        )
    _pending_restart_confirm.discard(chat_id)
    r = await client.post("/restart")
    if r.status_code == 200:
        await update.message.reply_text("♻️ Bot process restarted.")
    else:
        await update.message.reply_text(f"Restart failed: {r.text}")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/status /logs [n] /trades [n] /pnl [today|week|all] "
        "/balance /compound [on|off] /compound_status /withdraw <amount> "
        "/arb /markets /report [hour|session|daily] /backup "
        "/halt [reason] /resume /restart"
    )


def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("logs", logs_cmd))
    app.add_handler(CommandHandler("trades", trades_cmd))
    app.add_handler(CommandHandler("pnl", pnl_cmd))
    app.add_handler(CommandHandler("balance", balance_cmd))
    app.add_handler(CommandHandler("compound", compound_cmd))
    app.add_handler(CommandHandler("compound_status", compound_status_cmd))
    app.add_handler(CommandHandler("withdraw", withdraw_cmd))
    app.add_handler(CommandHandler("arb", arb_cmd))
    app.add_handler(CommandHandler("markets", markets_cmd))
    app.add_handler(CommandHandler("report", report_cmd))
    app.add_handler(CommandHandler("backup", backup_cmd))
    app.add_handler(CommandHandler("halt", halt_cmd))
    app.add_handler(CommandHandler("resume", resume_cmd))
    app.add_handler(CommandHandler("restart", restart_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.run_polling()


if __name__ == "__main__":
    main()