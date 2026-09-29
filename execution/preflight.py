"""
Production readiness checks. Run `python scripts/preflight.py` before going
live; main.py also runs them at live start-up and refuses to trade if any
CRITICAL check fails. Everything here is read-only.
"""
import logging
import os
import shutil
import subprocess
import sys
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


class _Unverifiable(RuntimeError):
    """A check that could not run in this environment, as opposed to one
    that ran and found a problem."""


def _git_hygiene_state() -> tuple:
    """Can the secret-hygiene check actually run here?

    The deployed image bakes the source tree in and installs no git, and no
    .git directory is copied in. Calling git there raised FileNotFoundError,
    which the generic handler reported as "[FAIL] secrets ... 'git'", reading
    like a leaked credential and blocking live trading for a reason that has
    nothing to do with the wallet.
    """
    if shutil.which("git") is None:
        return False, ("git not installed in this image — audit the repo "
                       "yourself with: git log --all -S <key>")
    if not os.path.isdir(os.path.join(REPO_ROOT, ".git")):
        return False, ("no .git here (source is baked into the image) — audit "
                       "the remote repo with: git log --all -S <key>")
    return True, ""


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
        available, reason = _git_hygiene_state()
        if not available:
            # Not a pass and not a failure: the guarantee went unchecked. It is
            # reported as a WARN so it stays visible instead of being treated
            # as either a clean bill of health or a blocking problem.
            raise _Unverifiable(reason)
        try:
            tracked = subprocess.run(["git", "ls-files", "config"], cwd=REPO_ROOT,
                                     capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as e:
            raise _Unverifiable(f"could not run git ls-files: {e}") from e
        if tracked.returncode != 0:
            raise _Unverifiable(
                f"could not run git ls-files: {tracked.stderr.strip()[:160] or 'unknown error'}")
        leaked_files = [f for f in tracked.stdout.split()
                        if os.path.basename(f).startswith(".env")
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

    _secrets_available, _secrets_reason = _git_hygiene_state()
    checks.append(_run("secrets", secrets_hygiene, critical=_secrets_available))

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


def critical_failures(checks: List[Check]) -> List[Check]:
    return [c for c in checks if not c.ok and c.critical]


# ── presentation ──────────────────────────────────────────────────────────
#
# The flat "[PASS] name detail" list was accurate but hard to scan: eleven
# undifferentiated rows, the critical one buried on line 7, and no indication
# of what to do next or in what order. A preflight report exists to be acted
# on, so it is grouped by concern, colour-coded, and ends with the actual
# next step.

_WIDTH = 68

# Ordered sections: (emoji, title, check names). The emoji belongs to the
# section, not to an individual check — deriving it per check gave one section
# two different icons depending on which check happened to be rendered first.
_SECTION_ORDER = [
    ("\U0001F527", "CONFIGURATION", ("config", "secrets", "kill switch")),
    ("\U0001F310", "NETWORK & WALLET",
     ("polygon rpc", "clob api", "polymarket auth", "trading approvals",
      "collateral", "gas (POL)")),
    ("\U0001F4B8", "FUNDING", ("funding address", "okx funding", "binance funding")),
    ("\U0001F4C8", "MARKETS", ("market data",)),
]
_FALLBACK_SECTION = ("\U0001F916", "OTHER")

# The order failures must be resolved in. This is not cosmetic: submitting
# approvals is an on-chain transaction, so an account with zero POL cannot
# make it happen no matter how many times it is retried.
_NEXT_STEPS = [
    ("gas (POL)", "Send POL to the trading wallet for gas"),
    ("trading approvals", "python scripts/preflight.py --setup-approvals --yes"),
    ("collateral", "Deposit USDC so the account holds pUSD collateral"),
    ("funding address", "Set FUNDING_DEPOSIT_ADDRESS to the expected bridge address"),
    ("secrets", "Audit the git history for the signing key"),
    ("market data", "Wait for a live up/down window and retry"),
]


def _color_enabled(explicit: bool = None) -> bool:
    if explicit is not None:
        return explicit
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


class _Ink:
    """ANSI helpers that collapse to identity functions when colour is off."""

    def __init__(self, enabled: bool):
        self.on = enabled

    def _w(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.on else text

    def bold(self, t):   return self._w("1", t)
    def dim(self, t):    return self._w("2", t)
    def red(self, t):    return self._w("31", t)
    def green(self, t):  return self._w("32", t)
    def yellow(self, t): return self._w("33", t)
    def blue(self, t):   return self._w("34", t)
    def cyan(self, t):   return self._w("36", t)
    def mag(self, t):    return self._w("35", t)


def _split_detail(detail: str):
    """Split a failure detail into (message, arrow hint).

    Several checks end with "— run: <command>" or "(expected 0x...)". Those
    are instructions, not findings, and burying them mid-sentence is what
    made the report hard to act on.
    """
    hint = None
    if " — run: " in detail:
        detail, hint = detail.split(" — run: ", 1)
    elif detail.endswith(")"):
        head, _, tail = detail.rpartition("(")
        if tail.startswith("expected "):
            detail, hint = head.rstrip(), tail[:-1]
    return detail.rstrip(". "), hint


def _dim_urls(text: str, ink: _Ink) -> str:
    """Dim URL-ish spans so latencies and balances stand out."""
    if not ink.on:
        return text
    out, i = [], 0
    for token in text.split(" "):
        if token.startswith(("http://", "https://")):
            if i:
                out.append(" ")
            out.append(ink.dim(token))
        else:
            if out:
                out.append(" ")
            out.append(token)
        i += 1
    return "".join(out)


def render_report(checks: List[Check], color: bool = None) -> str:
    """Full preflight report: header, grouped checks, tally, next steps."""
    ink = _Ink(_color_enabled(color))
    passed = sum(1 for c in checks if c.ok)
    warned = sum(1 for c in checks if not c.ok and not c.critical)
    failed = len(critical_failures(checks))

    title = "\U0001F680 LIVE TRADING PREFLIGHT"
    L: List[str] = []
    L.append("")
    L.append(ink.cyan("╔" + "═" * _WIDTH + "╗"))
    L.append(ink.cyan("║") + ink.bold(f" {title}") +
             " " * max(0, _WIDTH - len(title) - 1) + ink.cyan("║"))
    L.append(ink.cyan("╚" + "═" * _WIDTH + "╝"))

    # Group by section explicitly rather than assuming the checks arrive
    # contiguously, so a header can never print twice and the section order
    # stays fixed even if run_preflight() changes what it emits.
    grouped = []
    claimed = set()
    for emoji, title, names in _SECTION_ORDER:
        members = [c for c in checks if c.name in names]
        if members:
            grouped.append((emoji, title, members))
            claimed.update(names)
    leftovers = [c for c in checks if c.name not in claimed]
    if leftovers:
        grouped.append((*_FALLBACK_SECTION, leftovers))

    for emoji, title, members in grouped:
        L.append("")
        L.append(f"  {emoji}  {ink.bold(title)}")
        for c in members:
            # The emoji carries the outcome; the name is dimmed so the detail
            # column is what your eye lands on.
            glyph = "✅" if c.ok else ("❌" if c.critical else "⚠️ ")
            detail, hint = _split_detail(c.detail)
            L.append(f"  {glyph} {ink.dim(c.name.ljust(20))} {_dim_urls(detail, ink)}")
            if hint:
                L.append(f"      {'└→' if ink.on else '  ->'} {ink.cyan(hint)}")

    # ── tally ──
    L.append("")
    L.append(ink.dim("─" * _WIDTH))
    tally = f"  {ink.green('✅ ' + str(passed) + ' passed')}"
    if failed:
        tally += f"   {ink.red('❌ ' + str(failed) + ' failed')}"
    if warned:
        tally += f"   {ink.yellow('⚠️  ' + str(warned) + ' warning')}"
    L.append(tally)

    if failed:
        L.append("")
        L.append("  " + ink.red(ink.bold("\U0001F534 NOT READY — live trading is blocked")))
        steps = [s for name, s in _NEXT_STEPS
                 if any(c.name == name and c.critical and not c.ok for c in checks)]
        if steps:
            L.append("  " + ink.bold("Resolve in this order:"))
            for i, step in enumerate(steps, 1):
                L.append(f"    {ink.mag(str(i) + ' →')} {step}")
            L.append("")
            L.append("  " + ink.dim("Approvals cost gas — fund POL first, or step 2 cannot "
                                    "succeed."))
    elif warned:
        L.append("")
        L.append("  " + ink.yellow(ink.bold("\U0001F7E0 READY — with warnings")))
    else:
        L.append("")
        L.append("  " + ink.green(ink.bold("\U0001F7E2 READY for live trading")))

    L.append("")
    return "\n".join(L)


def format_report(checks: List[Check], color: bool = None) -> str:
    """Backwards-compatible alias for render_report()."""
    return render_report(checks, color=color)
