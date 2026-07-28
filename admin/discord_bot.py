"""
discord_bot.py — Discord front-end for the admin API.

Setup:
  1. https://discord.com/developers/applications -> New Application -> Bot -> copy token.
  2. Enable it, invite it to your server with the "applications.commands" +
     "bot" scopes (Send Messages, Use Slash Commands).
  3. Fill in .env: DISCORD_BOT_TOKEN, DISCORD_GUILD_ID, DISCORD_ALLOWED_ROLE_ID
     (or DISCORD_ALLOWED_USER_IDS if you don't want to bother with roles),
     ADMIN_API_URL, ADMIN_API_TOKEN.
  4. python3 discord_bot.py   (or run via systemd, see systemd/)

Slash commands mirror the Telegram bot: /status /logs /trades /pnl
/balance /compound /compound_status /withdraw /arb /markets
/halt /resume /restart
"""

import os

import discord
import httpx
from discord import app_commands

DISCORD_TOKEN = os.environ["DISCORD_BOT_TOKEN"]
GUILD_ID = int(os.environ["DISCORD_GUILD_ID"])
ALLOWED_ROLE_ID = os.environ.get("DISCORD_ALLOWED_ROLE_ID")
ALLOWED_USER_IDS = {
    int(x) for x in os.environ.get("DISCORD_ALLOWED_USER_IDS", "").split(",") if x.strip()
}
ADMIN_API_URL = os.environ.get("ADMIN_API_URL", "http://127.0.0.1:8787")
ADMIN_API_TOKEN = os.environ["ADMIN_API_TOKEN"]

HEADERS = {"Authorization": f"Bearer {ADMIN_API_TOKEN}"}
http_client = httpx.AsyncClient(base_url=ADMIN_API_URL, headers=HEADERS, timeout=15)

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

_pending_restart_confirm: set[int] = set()


def _authorized(interaction: discord.Interaction) -> bool:
    if interaction.user.id in ALLOWED_USER_IDS:
        return True
    if ALLOWED_ROLE_ID and isinstance(interaction.user, discord.Member):
        return any(str(r.id) == ALLOWED_ROLE_ID for r in interaction.user.roles)
    return False


async def _deny(interaction: discord.Interaction):
    await interaction.response.send_message("Not authorized for this command.", ephemeral=True)


@tree.command(name="status", description="Bot process + trading state", guild=discord.Object(id=GUILD_ID))
async def status_cmd(interaction: discord.Interaction):
    if not _authorized(interaction):
        return await _deny(interaction)
    r = await http_client.get("/status")
    d = r.json()
    halted = "🔴 HALTED" if d["trading_halted"] else "🟢 trading"
    msg = (
        f"**Process:** {d['process']}\n"
        f"**State:** {halted}"
        + (f" ({d['halt_reason']})" if d.get("halt_reason") else "")
        + f"\n**Last log:** {d.get('last_log_line') or '—'}"
    )
    await interaction.response.send_message(msg)


@tree.command(name="logs", description="Tail the bot's log file", guild=discord.Object(id=GUILD_ID))
@app_commands.describe(lines="how many lines (default 30, max 200)")
async def logs_cmd(interaction: discord.Interaction, lines: int = 30):
    if not _authorized(interaction):
        return await _deny(interaction)
    r = await http_client.get("/logs", params={"lines": lines})
    text = "\n".join(r.json()["lines"]) or "(empty)"
    await interaction.response.send_message(f"```\n{text[-1900:]}\n```")


@tree.command(name="trades", description="Recent trades", guild=discord.Object(id=GUILD_ID))
@app_commands.describe(limit="how many trades (default 10, max 50)")
async def trades_cmd(interaction: discord.Interaction, limit: int = 10):
    if not _authorized(interaction):
        return await _deny(interaction)
    r = await http_client.get("/trades", params={"limit": limit})
    rows = r.json()["trades"]
    if not rows:
        return await interaction.response.send_message("No trades found.")
    lines = [
        f"{t.get('timestamp','?')} {t.get('side','?'):>4} "
        f"${t.get('stake','?')} → {t.get('result','?')} ({t.get('pnl','?')})"
        for t in rows
    ]
    await interaction.response.send_message("```\n" + "\n".join(lines)[-1900:] + "\n```")


@tree.command(name="pnl", description="PnL summary", guild=discord.Object(id=GUILD_ID))
@app_commands.describe(period="today, week, or all")
@app_commands.choices(period=[
    app_commands.Choice(name="today", value="today"),
    app_commands.Choice(name="week", value="week"),
    app_commands.Choice(name="all", value="all"),
])
async def pnl_cmd(interaction: discord.Interaction, period: app_commands.Choice[str] = None):
    if not _authorized(interaction):
        return await _deny(interaction)
    p = period.value if period else "today"
    r = await http_client.get("/pnl", params={"period": p})
    d = r.json()
    await interaction.response.send_message(
        f"**PnL ({d['period']}):** {d['total_pnl']}\n"
        f"Trades: {d['trade_count']} — {d['wins']}W/{d['losses']}L "
        f"({d['win_rate_pct']}% win rate)"
    )


@tree.command(name="balance", description="Current bot balance snapshot", guild=discord.Object(id=GUILD_ID))
async def balance_cmd(interaction: discord.Interaction):
    if not _authorized(interaction):
        return await _deny(interaction)
    r = await http_client.get("/balance")
    if r.status_code != 200:
        return await interaction.response.send_message("Balance file not available.")
    await interaction.response.send_message(f"```\n{r.json()}\n```")


@tree.command(name="compound", description="Enable/disable profit compounding", guild=discord.Object(id=GUILD_ID))
@app_commands.describe(enabled="on or off")
@app_commands.choices(enabled=[
    app_commands.Choice(name="on", value="on"),
    app_commands.Choice(name="off", value="off"),
])
async def compound_cmd(interaction: discord.Interaction, enabled: app_commands.Choice[str] = None):
    if not _authorized(interaction):
        return await _deny(interaction)
    setting = enabled.value if enabled else "on"
    r = await http_client.post("/compound", json={"enabled": setting == "on"})
    if r.status_code == 200:
        state = "ENABLED" if setting == "on" else "DISABLED"
        await interaction.response.send_message(f"Profit compounding {state}")
    else:
        await interaction.response.send_message(f"Failed: {r.text}")


@tree.command(name="compound_status", description="Show compounding stats and growth rate", guild=discord.Object(id=GUILD_ID))
async def compound_status_cmd(interaction: discord.Interaction):
    if not _authorized(interaction):
        return await _deny(interaction)
    r = await http_client.get("/compound_status")
    if r.status_code != 200:
        return await interaction.response.send_message("Compound status not available.")
    d = r.json()
    await interaction.response.send_message(
        f"**Compounding:** {'ON' if d.get('enabled') else 'OFF'}\n"
        f"Principal: ${d.get('principal', 0):.2f}\n"
        f"Bankroll: ${d.get('bankroll', 0):.2f}\n"
        f"Profit: ${d.get('total_profit', 0):.2f} ({d.get('profit_pct', 0):.1f}%)\n"
        f"Withdrawn: ${d.get('total_withdrawn', 0):.2f}\n"
        f"Win Rate: {d.get('win_rate', 0):.1f}%\n"
        f"Trades: {d.get('total_trades', 0)} (W:{d.get('total_wins', 0)} L:{d.get('total_losses', 0)})"
    )


@tree.command(name="withdraw", description="Withdraw profits to wallet", guild=discord.Object(id=GUILD_ID))
@app_commands.describe(amount="amount in USD to withdraw")
async def withdraw_cmd(interaction: discord.Interaction, amount: float):
    if not _authorized(interaction):
        return await _deny(interaction)
    r = await http_client.post("/withdraw", json={"amount_usd": amount})
    if r.status_code == 200:
        d = r.json()
        await interaction.response.send_message(
            f"Withdrawal requested: ${amount:.2f}\n"
            f"Status: {d.get('status', 'pending')}\n"
            f"Remaining profit: ${d.get('remaining_profit', 0):.2f}"
        )
    else:
        await interaction.response.send_message(f"Withdrawal failed: {r.text}")


@tree.command(name="arb", description="Show arbitrage opportunities and PnL", guild=discord.Object(id=GUILD_ID))
async def arb_cmd(interaction: discord.Interaction):
    if not _authorized(interaction):
        return await _deny(interaction)
    r = await http_client.get("/arb")
    if r.status_code != 200:
        return await interaction.response.send_message("Arb data not available.")
    d = r.json()
    await interaction.response.send_message(
        f"**Arbitrage Stats:**\n"
        f"Bankroll: ${d.get('bankroll', 0):.2f}\n"
        f"Open Positions: {d.get('open_positions', 0)}\n"
        f"Total PnL: ${d.get('total_pnl', 0):.4f}\n"
        f"Win Rate: {d.get('win_rate', 0):.1f}%\n"
        f"PM Markets: {d.get('pm_markets_tracked', 0)}"
    )


@tree.command(name="markets", description="Show active 5-minute market windows", guild=discord.Object(id=GUILD_ID))
async def markets_cmd(interaction: discord.Interaction):
    if not _authorized(interaction):
        return await _deny(interaction)
    r = await http_client.get("/markets")
    if r.status_code != 200:
        return await interaction.response.send_message("Market data not available.")
    d = r.json()
    markets = d.get("markets", [])
    if not markets:
        return await interaction.response.send_message("No active market windows.")
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
    await interaction.response.send_message("\n".join(lines))


@tree.command(name="report", description="Hourly/session/daily trade report", guild=discord.Object(id=GUILD_ID))
@app_commands.describe(period="report period")
@app_commands.choices(period=[
    app_commands.Choice(name="hour", value="hour"),
    app_commands.Choice(name="session", value="session"),
    app_commands.Choice(name="daily", value="daily"),
])
async def report_cmd(interaction: discord.Interaction, period: app_commands.Choice[str] = None):
    if not _authorized(interaction):
        return await _deny(interaction)
    p = period.value if period else "hour"
    r = await http_client.get(f"/report?period={p}")
    if r.status_code != 200:
        return await interaction.response.send_message("Report not available.")
    d = r.json()
    if d.get("total_trades", 0) == 0 and p == "hour":
        return await interaction.response.send_message("No trades in the last hour.")
    label = "Session" if p == "session" else "Hourly" if p == "hour" else "Daily"
    lines = [
        f"**{label} Report:**\n",
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
    await interaction.response.send_message("\n".join(lines))


@tree.command(name="backup", description="Trigger a manual database backup", guild=discord.Object(id=GUILD_ID))
async def backup_cmd(interaction: discord.Interaction):
    if not _authorized(interaction):
        return await _deny(interaction)
    r = await http_client.post("/backup")
    if r.status_code == 200:
        d = r.json()
        await interaction.response.send_message(f"Backup completed: {d.get('backup_path', 'unknown')}")
    else:
        await interaction.response.send_message(f"Backup failed: {r.text}")


@tree.command(name="halt", description="Pause trading", guild=discord.Object(id=GUILD_ID))
@app_commands.describe(reason="why you're halting it")
async def halt_cmd(interaction: discord.Interaction, reason: str = "manual halt via discord"):
    if not _authorized(interaction):
        return await _deny(interaction)
    await http_client.post("/halt", json={"reason": reason, "actor": f"discord:{interaction.user.id}"})
    await interaction.response.send_message(f"🔴 Trading halted. Reason: {reason}")


@tree.command(name="resume", description="Resume trading", guild=discord.Object(id=GUILD_ID))
async def resume_cmd(interaction: discord.Interaction):
    if not _authorized(interaction):
        return await _deny(interaction)
    await http_client.post("/resume", json={"actor": f"discord:{interaction.user.id}"})
    await interaction.response.send_message("🟢 Trading resumed.")


@tree.command(name="restart", description="Hard restart the bot process (asks for confirmation)", guild=discord.Object(id=GUILD_ID))
async def restart_cmd(interaction: discord.Interaction):
    if not _authorized(interaction):
        return await _deny(interaction)
    uid = interaction.user.id
    if uid not in _pending_restart_confirm:
        _pending_restart_confirm.add(uid)
        return await interaction.response.send_message(
            "This hard-restarts the bot process (not just a pause). "
            "Run /restart again within 30s to confirm.", ephemeral=True
        )
    _pending_restart_confirm.discard(uid)
    r = await http_client.post("/restart")
    if r.status_code == 200:
        await interaction.response.send_message("♻️ Bot process restarted.")
    else:
        await interaction.response.send_message(f"Restart failed: {r.text}")


@client.event
async def on_ready():
    await tree.sync(guild=discord.Object(id=GUILD_ID))
    print(f"Logged in as {client.user}")


if __name__ == "__main__":
    client.run(DISCORD_TOKEN)