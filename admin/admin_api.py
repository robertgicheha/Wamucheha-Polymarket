"""
admin_api.py — Internal control-plane API for the Polymarket BTC bot.

Both the Telegram bot and the Discord bot talk to THIS service instead of
touching the trading bot's files/DB/process directly. That keeps one
auth boundary, one set of guardrails, and one audit log, regardless of
how many chat surfaces you add later.

Run it on the same VPS as the trading bot:
    uvicorn admin_api:app --host 127.0.0.1 --port 8787

It should NOT be exposed to the public internet — bind to 127.0.0.1 and
let the Telegram/Discord bot processes (also on the VPS, or reached via
an SSH tunnel) call it locally. See systemd/ and README.md.

ASSUMPTIONS ABOUT YOUR TRADING BOT (adjust the CONFIG section below to match):
  1. Trades are logged as rows in a SQLite DB, one row per bet, with at
     least: timestamp, market, side, stake, entry_price, result, pnl.
  2. The bot writes a rolling log file (plain text, newest lines at bottom).
  3. The bot polls a small JSON "control file" on every trading loop
     iteration (every few seconds is fine) and skips trading if
     control["halted"] is true. This is the safe way to pause it —
     far safer than restarting the process mid-trade.
  4. The bot runs under systemd as its own unit, so /restart can call
     `systemctl restart <unit>` for the rare case you actually need
     a hard restart (e.g. it's frozen, not just "stop trading").

If your bot doesn't yet write to a DB or check a control file, see the
"MINIMAL BOT-SIDE HOOKS" section at the bottom of README.md — it's a
handful of lines, not a refactor.
"""

import json
import os
import sqlite3
import subprocess
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ── CONFIG — adjust these to match your actual bot's paths/schema ──────────
ADMIN_TOKEN = os.environ["ADMIN_API_TOKEN"]  # shared secret, set in .env
BOT_DB_PATH = os.environ.get("BOT_DB_PATH", "/opt/btc-bot/data/trades.db")
BOT_LOG_PATH = os.environ.get("BOT_LOG_PATH", "/opt/btc-bot/logs/bot.log")
CONTROL_FILE = os.environ.get("CONTROL_FILE_PATH", "/opt/btc-bot/data/control.json")
BALANCE_FILE = os.environ.get("BALANCE_FILE_PATH", "/opt/btc-bot/data/balance.json")
BOT_SYSTEMD_UNIT = os.environ.get("BOT_SYSTEMD_UNIT", "btc-bot.service")
TRADES_TABLE = os.environ.get("TRADES_TABLE_NAME", "trades")

app = FastAPI(title="Polymarket BTC Bot — Admin API")


def check_auth(authorization: Optional[str]):
    if authorization != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")


@contextmanager
def db():
    if not Path(BOT_DB_PATH).exists():
        raise HTTPException(status_code=503, detail=f"trades DB not found at {BOT_DB_PATH}")
    conn = sqlite3.connect(BOT_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def read_control() -> dict:
    p = Path(CONTROL_FILE)
    if not p.exists():
        return {"halted": False, "reason": None, "changed_at": None}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {"halted": False, "reason": None, "changed_at": None}


def write_control(halted: bool, reason: Optional[str], actor: str):
    Path(CONTROL_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(CONTROL_FILE).write_text(json.dumps({
        "halted": halted,
        "reason": reason,
        "changed_at": datetime.now(timezone.utc).isoformat(),
        "changed_by": actor,
    }, indent=2))


# ── Schemas ──────────────────────────────────────────────────────────────
class HaltRequest(BaseModel):
    reason: str = "manual halt"
    actor: str = "unknown"


class ResumeRequest(BaseModel):
    actor: str = "unknown"


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/status")
def status(authorization: str = Header(None)):
    check_auth(authorization)
    control = read_control()
    try:
        proc = subprocess.run(
            ["systemctl", "is-active", BOT_SYSTEMD_UNIT],
            capture_output=True, text=True, timeout=5,
        )
        process_state = proc.stdout.strip() or proc.stderr.strip()
    except Exception as e:
        process_state = f"unknown ({e})"

    last_log_line = None
    if Path(BOT_LOG_PATH).exists():
        with open(BOT_LOG_PATH, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4000))
            tail = f.read().decode(errors="ignore").strip().splitlines()
            last_log_line = tail[-1] if tail else None

    return {
        "process": process_state,
        "trading_halted": control.get("halted", False),
        "halt_reason": control.get("reason"),
        "halt_changed_at": control.get("changed_at"),
        "last_log_line": last_log_line,
    }


@app.get("/logs")
def logs(lines: int = 50, authorization: str = Header(None)):
    check_auth(authorization)
    lines = max(1, min(lines, 500))
    if not Path(BOT_LOG_PATH).exists():
        raise HTTPException(status_code=404, detail="log file not found")
    with open(BOT_LOG_PATH, "r", errors="ignore") as f:
        all_lines = f.readlines()
    return {"lines": [l.rstrip("\n") for l in all_lines[-lines:]]}


@app.get("/trades")
def trades(limit: int = 20, authorization: str = Header(None)):
    check_auth(authorization)
    limit = max(1, min(limit, 200))
    with db() as conn:
        rows = conn.execute(
            f"SELECT * FROM {TRADES_TABLE} ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return {"trades": [dict(r) for r in rows]}


@app.get("/pnl")
def pnl(period: str = "today", authorization: str = Header(None)):
    check_auth(authorization)
    if period not in ("today", "week", "all"):
        raise HTTPException(status_code=400, detail="period must be today|week|all")

    since = None
    now = datetime.now(timezone.utc)
    if period == "today":
        since = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        since = now - timedelta(days=7)

    with db() as conn:
        if since:
            rows = conn.execute(
                f"SELECT pnl, result FROM {TRADES_TABLE} WHERE timestamp >= ?",
                (since.isoformat(),),
            ).fetchall()
        else:
            rows = conn.execute(f"SELECT pnl, result FROM {TRADES_TABLE}").fetchall()

    total = sum(r["pnl"] for r in rows if r["pnl"] is not None)
    wins = sum(1 for r in rows if (r["result"] or "").lower() == "win")
    losses = sum(1 for r in rows if (r["result"] or "").lower() == "loss")
    n = len(rows)
    win_rate = (wins / n * 100) if n else 0.0

    return {
        "period": period,
        "trade_count": n,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate, 1),
        "total_pnl": round(total, 2),
    }


@app.get("/balance")
def balance(authorization: str = Header(None)):
    check_auth(authorization)
    if not Path(BALANCE_FILE).exists():
        raise HTTPException(status_code=404, detail="balance file not found — have the bot write balance.json")
    return json.loads(Path(BALANCE_FILE).read_text())


@app.post("/halt")
def halt(req: HaltRequest, authorization: str = Header(None)):
    check_auth(authorization)
    write_control(True, req.reason, req.actor)
    return {"ok": True, "halted": True, "reason": req.reason}


@app.post("/resume")
def resume(req: ResumeRequest, authorization: str = Header(None)):
    check_auth(authorization)
    write_control(False, None, req.actor)
    return {"ok": True, "halted": False}


@app.post("/restart")
def restart(authorization: str = Header(None)):
    """Hard restart of the bot process via systemd. Use sparingly — prefer
    /halt + /resume, which pause trading without dropping in-flight state.
    Requires a sudoers rule (see README) so this can run without a shell login."""
    check_auth(authorization)
    try:
        subprocess.run(
            ["sudo", "/usr/bin/systemctl", "restart", BOT_SYSTEMD_UNIT],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"restart failed: {e.stderr}")
    return {"ok": True, "restarted": BOT_SYSTEMD_UNIT}