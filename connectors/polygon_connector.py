"""
Polygon on-chain connector. Handles wallet balance checks, USDC.e transfers
for funding/withdrawal, and reading on-chain CTF Exchange events.
Uses web3.py.

Install: pip install web3
"""
import logging
from typing import Optional

from web3 import Web3

from config.settings import settings

logger = logging.getLogger(__name__)

# USDC.e on Polygon mainnet
USDC_E_ADDRESS = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
# MATIC (WMATIC wrapped for balance check via native)
WMATIC_ADDRESS = Web3.to_checksum_address("0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270")

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
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
]

# Minimal ABI for native MATIC balance
NATIVE_BALANCE_ABI = []  # web3.py handles native balance natively

# CTF Exchange contract event ABIs for reading trade history
CTF_EXCHANGE_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": " maker", "type": "address"},
            {"indexed": True, "name": " taker", "type": "address"},
            {"indexed": True, "name": " tokenId", "type": "uint256"},
            {"indexed": False, "name": " makerAmount", "type": "uint256"},
            {"indexed": False, "name": " takerAmount", "type": "uint256"},
            {"indexed": False, "name": " side", "type": "uint8"},
        ],
        "name": "Trade",
        "type": "event",
    },
]


class PolygonConnector:
    def __init__(self):
        self.rpc_url = settings.polygon_rpc_url
        self.wallet_address = (
            Web3.to_checksum_address(settings.polygon_wallet_address)
            if settings.polygon_wallet_address
            else None
        )
        self._w3: Optional[Web3] = None
        self._usdc_contract = None

    def _get_web3(self) -> Web3:
        if self._w3 is None:
            self._w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            if not self._w3.is_connected():
                logger.error("Failed to connect to Polygon RPC at %s", self.rpc_url)
                raise ConnectionError(f"Cannot connect to Polygon RPC: {self.rpc_url}")
        return self._w3

    def _get_usdc_contract(self):
        if self._usdc_contract is None:
            w3 = self._get_web3()
            self._usdc_contract = w3.eth.contract(
                address=USDC_E_ADDRESS, abi=ERC20_ABI
            )
        return self._usdc_contract

    def get_usdc_balance(self, address: Optional[str] = None) -> float:
        """Query USDC.e balanceOf(address), convert from 6-decimal units."""
        addr = address or self.wallet_address
        if not addr:
            raise ValueError("No wallet address configured or provided")
        checksum = Web3.to_checksum_address(addr)
        contract = self._get_usdc_contract()
        raw_balance = contract.functions.balanceOf(checksum).call()
        return raw_balance / (10 ** 6)  # USDC.e has 6 decimals

    def get_matic_balance(self, address: Optional[str] = None) -> float:
        """Query native MATIC balance, convert from 18-decimal units."""
        addr = address or self.wallet_address
        if not addr:
            raise ValueError("No wallet address configured or provided")
        checksum = Web3.to_checksum_address(addr)
        w3 = self._get_web3()
        raw_balance = w3.eth.get_balance(checksum)
        return w3.from_wei(raw_balance, "ether")

    def get_balances(self, address: Optional[str] = None) -> dict:
        """Get both MATIC and USDC.e balances."""
        return {
            "matic": self.get_matic_balance(address),
            "usdc": self.get_usdc_balance(address),
        }

    def transfer_usdc(self, to_address: str, amount_usd: float) -> str:
        """
        Build, sign, and broadcast a USDC.e transfer transaction.
        Used for the withdrawal flow (moving profit to a cold wallet).
        Returns tx hash.
        """
        if settings.trading_mode != "live":
            raise RuntimeError("transfer_usdc called while not in live mode")

        w3 = self._get_web3()
        contract = self._get_usdc_contract()
        checksum_to = Web3.to_checksum_address(to_address)
        raw_amount = int(amount_usd * (10 ** 6))

        nonce = w3.eth.get_transaction_count(
            Web3.to_checksum_address(settings.polygon_wallet_address)
        )

        # Estimate gas
        gas_estimate = contract.functions.transfer(
            checksum_to, raw_amount
        ).estimate_gas({
            "from": Web3.to_checksum_address(settings.polygon_wallet_address)
        })

        tx = contract.functions.transfer(checksum_to, raw_amount).build_transaction({
            "chainId": 137,
            "gas": int(gas_estimate * 1.2),
            "gasPrice": w3.eth.gas_price,
            "nonce": nonce,
        })

        from web3.middleware import geth_poa_middleware
        # Sign and send
        signed = w3.eth.account.sign_transaction(
            tx, private_key=settings.polymarket_private_key
        )
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        if receipt.status != 1:
            raise RuntimeError(f"USDC transfer failed: tx {tx_hash.hex()}")

        logger.info("USDC transfer succeeded: %s", tx_hash.hex())
        return tx_hash.hex()

    def check_gas_balance(self) -> dict:
        """
        Check if MATIC balance is sufficient for gas. Returns balance and
        a warning if below threshold.
        """
        matic = self.get_matic_balance()
        threshold = 0.1  # MATIC — roughly enough for ~50-100 simple transactions
        return {
            "balance_matic": matic,
            "sufficient": matic >= threshold,
            "warning": (
                f"Low MATIC balance ({matic:.4f}) — may not have enough for gas. "
                f"Top up your Polygon wallet."
                if matic < threshold
                else None
            ),
        }

    def get_polygon_block_number(self) -> int:
        """Get the latest block number for indexing purposes."""
        return self._get_web3().eth.block_number

    def get_block_timestamp(self, block_number: int) -> int:
        """Get the timestamp of a specific block."""
        block = self._get_web3().eth.get_block(block_number)
        return block["timestamp"]
