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
        "/balance /halt [reason] /resume /restart"
    )


def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("logs", logs_cmd))
    app.add_handler(CommandHandler("trades", trades_cmd))
    app.add_handler(CommandHandler("pnl", pnl_cmd))
    app.add_handler(CommandHandler("balance", balance_cmd))
    app.add_handler(CommandHandler("halt", halt_cmd))
    app.add_handler(CommandHandler("resume", resume_cmd))
    app.add_handler(CommandHandler("restart", restart_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.run_polling()


if __name__ == "__main__":
    main()