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
/balance /halt /resume /restart
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