"""
Polygon on-chain connector: RPC health, wallet balances (pUSD, USDC, USDC.e,
POL) and ERC-20 transfers for the withdrawal flow. Uses web3.py >= 7.

RPC resilience: POLYGON_RPC_URL is tried first, then each entry of
POLYGON_RPC_FALLBACK_URLS. A provider is only accepted if it reports chain
id 137 and a recent block, and a failing provider is swapped out on the next
call instead of wedging the bot.

Install: pip install web3
"""
import logging
import re
import threading
import time
from typing import Callable, Dict, List, Optional, TypeVar

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

from config.settings import settings

logger = logging.getLogger(__name__)

POLYGON_CHAIN_ID = 137
MAX_BLOCK_AGE_SECONDS = 60

# pUSD is Polymarket's trading collateral since the 2026-04-28 upgrade.
PUSD_ADDRESS = Web3.to_checksum_address("0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB")
USDC_ADDRESS = Web3.to_checksum_address("0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359")
USDC_E_ADDRESS = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
TOKENS = {"pusd": PUSD_ADDRESS, "usdc": USDC_ADDRESS, "usdc_e": USDC_E_ADDRESS}
TOKEN_DECIMALS = 6

ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"},
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
]

T = TypeVar("T")


def redact_rpc_url(url: str) -> str:
    """Hide API keys embedded in RPC URLs before logging them."""
    return re.sub(r"/[A-Za-z0-9_-]{16,}", "/<key>", url or "")


class PolygonConnector:
    def __init__(self, rpc_urls: Optional[List[str]] = None):
        urls = rpc_urls if rpc_urls is not None else (
            [settings.polygon_rpc_url] + list(settings.polygon_rpc_fallback_urls)
        )
        self.rpc_urls = [u for i, u in enumerate(urls) if u and u not in urls[:i]]
        self.wallet_address = (
            Web3.to_checksum_address(settings.polygon_wallet_address)
            if settings.polygon_wallet_address
            else None
        )
        self._w3: Optional[Web3] = None
        self._active_url: Optional[str] = None
        self._lock = threading.Lock()

    # ── Connection management ─────────────────────────────────────────

    def _connect(self, url: str) -> Web3:
        w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": settings.polygon_rpc_timeout_seconds}))
        # Polygon blocks carry >32 bytes of extraData; without this middleware
        # every get_block() call raises ExtraDataLengthError.
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        chain_id = w3.eth.chain_id
        if chain_id != POLYGON_CHAIN_ID:
            raise ConnectionError(f"wrong chain id {chain_id} (expected {POLYGON_CHAIN_ID})")
        age = time.time() - w3.eth.get_block("latest")["timestamp"]
        if age > MAX_BLOCK_AGE_SECONDS:
            raise ConnectionError(f"stale node: latest block is {age:.0f}s old")
        return w3

    def _get_web3(self) -> Web3:
        with self._lock:
            if self._w3 is not None:
                return self._w3
            errors = []
            for url in self.rpc_urls:
                try:
                    self._w3 = self._connect(url)
                    self._active_url = url
                    logger.info("Polygon RPC connected: %s", redact_rpc_url(url))
                    return self._w3
                except Exception as e:
                    errors.append(f"{redact_rpc_url(url)}: {e}")
                    logger.warning("Polygon RPC unavailable: %s — %s", redact_rpc_url(url), e)
            raise ConnectionError("No healthy Polygon RPC: " + "; ".join(errors))

    def _call(self, fn: Callable[[Web3], T]) -> T:
        """Run a read against the active RPC, failing over once on error."""
        try:
            return fn(self._get_web3())
        except Exception as e:
            logger.warning("Polygon RPC call failed on %s (%s), failing over",
                           redact_rpc_url(self._active_url or ""), e)
            with self._lock:
                failed = self._active_url
                self._w3 = None
                if failed in self.rpc_urls and len(self.rpc_urls) > 1:
                    self.rpc_urls.remove(failed)
                    self.rpc_urls.append(failed)
            return fn(self._get_web3())

    @property
    def active_rpc(self) -> str:
        return redact_rpc_url(self._active_url or "")

    def health(self) -> Dict:
        """Latency / block freshness of every configured RPC."""
        report = []
        for url in self.rpc_urls:
            t0 = time.time()
            try:
                w3 = self._connect(url)
                block = w3.eth.get_block("latest")
                report.append({
                    "url": redact_rpc_url(url), "ok": True,
                    "latency_ms": round((time.time() - t0) * 1000),
                    "block": block["number"],
                    "block_age_s": round(time.time() - block["timestamp"], 1),
                    "base_fee_gwei": round(block.get("baseFeePerGas", 0) / 1e9, 1),
                })
            except Exception as e:
                report.append({"url": redact_rpc_url(url), "ok": False, "error": str(e)[:200]})
        return {"rpcs": report, "healthy": any(r["ok"] for r in report)}

    # ── Balances ──────────────────────────────────────────────────────

    def _resolve(self, address: Optional[str]) -> str:
        addr = address or self.wallet_address
        if not addr:
            raise ValueError("No wallet address configured or provided")
        return Web3.to_checksum_address(addr)

    def get_token_balance(self, token: str, address: Optional[str] = None) -> float:
        addr = self._resolve(address)
        token_address = TOKENS[token]
        raw = self._call(
            lambda w3: w3.eth.contract(address=token_address, abi=ERC20_ABI)
            .functions.balanceOf(addr).call()
        )
        return raw / 10 ** TOKEN_DECIMALS

    def get_usdc_balance(self, address: Optional[str] = None) -> float:
        """Trading collateral (pUSD) balance. Kept for backwards compatibility."""
        return self.get_token_balance("pusd", address)

    def get_pol_balance(self, address: Optional[str] = None) -> float:
        addr = self._resolve(address)
        return self._call(lambda w3: w3.eth.get_balance(addr)) / 1e18

    # Pre-2024 name for POL
    get_matic_balance = get_pol_balance

    def get_balances(self, address: Optional[str] = None) -> Dict[str, float]:
        """POL plus every stablecoin the bot cares about."""
        balances = {"pol": self.get_pol_balance(address)}
        for token in TOKENS:
            balances[token] = self.get_token_balance(token, address)
        return balances

    def is_contract(self, address: str) -> bool:
        addr = Web3.to_checksum_address(address)
        return len(self._call(lambda w3: w3.eth.get_code(addr))) > 0

    def check_gas_balance(self, threshold: float = 0.5) -> Dict:
        """
        Only EOA wallets pay gas themselves; Deposit/Proxy/Safe wallets trade
        and redeem gaslessly through Polymarket's relayer.
        """
        pol = self.get_pol_balance()
        return {
            "balance_pol": pol,
            "sufficient": pol >= threshold,
            "warning": (
                f"Low POL balance ({pol:.4f}) — an EOA wallet needs POL for gas."
                if pol < threshold else None
            ),
        }

    # ── Transfers ─────────────────────────────────────────────────────

    def transfer_token(self, to_address: str, amount_usd: float, token: str = "pusd") -> str:
        """
        Sign and broadcast an ERC-20 transfer from the EOA signer (the
        withdrawal flow for EOA wallets). Returns the tx hash.
        """
        if settings.trading_mode != "live":
            raise RuntimeError("transfer_token called while not in live mode")
        if amount_usd <= 0:
            raise ValueError("amount must be positive")

        from eth_account import Account

        w3 = self._get_web3()
        sender = Account.from_key(settings.polymarket_private_key).address
        contract = w3.eth.contract(address=TOKENS[token], abi=ERC20_ABI)
        checksum_to = Web3.to_checksum_address(to_address)
        raw_amount = int(round(amount_usd * 10 ** TOKEN_DECIMALS))

        balance = contract.functions.balanceOf(sender).call()
        if balance < raw_amount:
            raise RuntimeError(
                f"insufficient {token}: have {balance / 1e6:.2f}, need {amount_usd:.2f}"
            )

        fn = contract.functions.transfer(checksum_to, raw_amount)
        gas = fn.estimate_gas({"from": sender})
        base_fee = w3.eth.get_block("latest")["baseFeePerGas"]
        priority = w3.eth.max_priority_fee
        tx = fn.build_transaction({
            "chainId": POLYGON_CHAIN_ID,
            "from": sender,
            "gas": int(gas * 1.2),
            "maxPriorityFeePerGas": priority,
            "maxFeePerGas": base_fee * 2 + priority,
            "nonce": w3.eth.get_transaction_count(sender, "pending"),
        })
        signed = w3.eth.account.sign_transaction(tx, private_key=settings.polymarket_private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        if receipt.status != 1:
            raise RuntimeError(f"{token} transfer failed: tx {tx_hash.hex()}")
        logger.info("%s transfer of $%.2f succeeded: %s", token, amount_usd, tx_hash.hex())
        return tx_hash.hex()

    def transfer_usdc(self, to_address: str, amount_usd: float) -> str:
        """Backwards-compatible name: transfers pUSD (the trading collateral)."""
        return self.transfer_token(to_address, amount_usd, "pusd")

    def get_polygon_block_number(self) -> int:
        """Get the latest block number for indexing purposes."""
        return self._call(lambda w3: w3.eth.block_number)

    def get_block_timestamp(self, block_number: int) -> int:
        """Get the timestamp of a specific block."""
        return self._call(lambda w3: w3.eth.get_block(block_number)["timestamp"])
