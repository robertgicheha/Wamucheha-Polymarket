"""
Production readiness checks. Run `python scripts/preflight.py` before going
live; main.py also runs them at live start-up and refuses to trade if any
CRITICAL check fails. Everything here is read-only.
"""
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, List

from config.settings import settings

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    critical: bool = True


def _run(name: str, fn: Callable[[], str], critical: bool = True) -> Check:
    try:
        return Check(name, True, fn() or "ok", critical)
    except Exception as e:
        return Check(name, False, str(e)[:300], critical)


def _key_in_git_history(secret: str) -> bool:
    """True if `secret` appears in any commit of this repository."""
    needle = secret[2:] if secret.startswith("0x") else secret
    if len(needle) < 32:
        return False
    try:
        out = subprocess.run(
            ["git", "log", "--all", "--format=%h", "-S", needle],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def run_preflight(include_exchanges: bool = True) -> List[Check]:
    from connectors.polygon_connector import PolygonConnector
    from connectors.polymarket_connector import PolymarketConnector

    checks: List[Check] = []
    pm = PolymarketConnector()

    def config():
        settings.validate_for_live_trading()
        return "live-trading config complete"
    checks.append(_run("config", config))

    def secrets_hygiene():
        tracked = subprocess.run(["git", "ls-files", "config"], cwd=REPO_ROOT,
                                 capture_output=True, text=True).stdout.split()
        leaked_files = [f for f in tracked if os.path.basename(f).startswith(".env")
                        and not f.endswith(".env.example")]
        if leaked_files:
            raise RuntimeError(f"secret files tracked by git: {leaked_files}")
        for label, secret in (("POLYMARKET_PRIVATE_KEY", settings.polymarket_private_key),
                              ("GNOSIS/PMXT key", settings.pmxt_private_key)):
            if secret and _key_in_git_history(secret):
                raise RuntimeError(
                    f"{label} appears in git history (and on GitHub if pushed) — treat it as "
                    f"compromised: create a new wallet and never fund the old one"
                )
        return "no secrets tracked; signing key not in git history"
    checks.append(_run("secrets", secrets_hygiene))

    def kill_switch():
        if os.path.exists(settings.kill_switch_file):
            raise RuntimeError(f"kill switch file '{settings.kill_switch_file}' present")
        return "not engaged"
    checks.append(_run("kill switch", kill_switch))

    polygon = PolygonConnector()

    def rpc():
        report = polygon.health()
        lines = [
            f"{r['url']}: " + (f"{r['latency_ms']}ms, block age {r['block_age_s']}s" if r["ok"]
                               else f"DOWN ({r['error'][:80]})")
            for r in report["rpcs"]
        ]
        if not report["rpcs"] or not report["rpcs"][0]["ok"]:
            if report["healthy"]:
                return "PRIMARY DOWN, using fallback — " + "; ".join(lines)
            raise RuntimeError("no healthy RPC — " + "; ".join(lines))
        return "; ".join(lines)
    checks.append(_run("polygon rpc", rpc))

    def clob():
        version = pm.get_clob_version()
        if version != 2:
            raise RuntimeError(f"CLOB version {version}, this bot targets V2")
        return "CLOB V2 reachable"
    checks.append(_run("clob api", clob))

    wallet_info = {}

    def auth():
        client = pm.secure_client()
        wallet_info["wallet"] = client.wallet
        wallet_info["type"] = str(client.wallet_type)
        if settings.polymarket_funder_address and \
                client.wallet.lower() != settings.polymarket_funder_address.lower():
            raise RuntimeError(f"SDK resolved wallet {client.wallet} != POLYMARKET_FUNDER_ADDRESS")
        return f"wallet {client.wallet} ({client.wallet_type})"
    checks.append(_run("polymarket auth", auth))

    if wallet_info:
        def approvals():
            if not pm.trading_approvals_ready():
                raise RuntimeError("missing approvals — run: python scripts/preflight.py --setup-approvals")
            return "all approvals set"
        checks.append(_run("trading approvals", approvals))

        def collateral():
            bal = pm.get_collateral_balance()
            if bal < 5:
                raise RuntimeError(f"pUSD balance ${bal:.2f} — fund the account (min order ≈ $2.50–5)")
            return f"${bal:.2f} pUSD available"
        checks.append(_run("collateral", collateral))

        if "EOA" in wallet_info["type"].upper():
            def gas():
                polygon.wallet_address = wallet_info["wallet"]
                g = polygon.check_gas_balance()
                if not g["sufficient"]:
                    raise RuntimeError(g["warning"])
                return f"{g['balance_pol']:.3f} POL"
            checks.append(_run("gas (POL)", gas))

        if settings.funding_deposit_address:
            def deposit_address():
                from execution.funding import get_bridge_evm_address
                expected = get_bridge_evm_address(wallet_info["wallet"])
                if not expected or expected.lower() != settings.funding_deposit_address.lower():
                    raise RuntimeError(f"FUNDING_DEPOSIT_ADDRESS is not this wallet's bridge "
                                       f"address (expected {expected})")
                return "matches Polymarket bridge address for this wallet"
            checks.append(_run("funding address", deposit_address))

    def market_data():
        now = int(time.time())
        interval = settings.updown_interval_minutes * 60
        found = []
        for asset in settings.updown_assets:
            m = pm.get_updown_market(asset, settings.updown_interval_minutes, now - now % interval)
            if m is None:
                raise RuntimeError(f"no current {asset} up/down window found")
            books = pm.get_books([m.token_id_up, m.token_id_down])
            if len(books) != 2:
                raise RuntimeError(f"order books unavailable for {m.slug}")
            fees = pm.get_fee_params(m.condition_id)
            found.append(f"{m.slug} (taker fee {fees.rate}·(p(1-p))^{fees.exponent})")
        return "; ".join(found)
    checks.append(_run("market data", market_data))

    if include_exchanges:
        if settings.okx_api_key:
            def okx():
                from connectors.okx_connector import OKXConnector
                o = OKXConnector()
                perms = o.get_api_permissions()
                chain = o.get_polygon_usdc_chain()
                bal = o.get_usdc_balance()
                if "withdraw" not in perms["perm"]:
                    raise RuntimeError(f"OKX key lacks withdraw permission (perm={perms['perm']})")
                return f"perm={perms['perm']} chain={chain['chain']} fee={chain['fee']} USDC=${bal:.2f}"
            checks.append(_run("okx funding", okx, critical=settings.auto_funding_enabled
                               and settings.funding_source == "okx"))
        if settings.binance_api_key:
            def binance():
                from connectors.binance_connector import BinanceConnector
                b = BinanceConnector()
                perms = b.get_api_permissions()
                net = b.get_polygon_usdc_network()
                bal = b.get_usdc_balance()
                if not perms["can_withdraw"]:
                    raise RuntimeError("Binance key has withdrawals disabled (needs IP restriction + "
                                       "'Enable Withdrawals')")
                return f"network={net['network']} fee={net['fee']} USDC=${bal:.2f}"
            checks.append(_run("binance funding", binance, critical=settings.auto_funding_enabled
                               and settings.funding_source == "binance"))
    return checks


def format_report(checks: List[Check]) -> str:
    lines = []
    for c in checks:
        status = "PASS" if c.ok else ("FAIL" if c.critical else "WARN")
        lines.append(f"[{status}] {c.name:18s} {c.detail}")
    return "\n".join(lines)


def critical_failures(checks: List[Check]) -> List[Check]:
    return [c for c in checks if not c.ok and c.critical]
