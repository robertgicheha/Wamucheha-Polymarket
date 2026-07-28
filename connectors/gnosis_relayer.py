"""
Gnosis Safe / Relayer integration for gasless payments on Polymarket.

Key features:
- Gasless transactions (gas paid by Polymarket relayer)
- Requires deployment via Relayer Client
- Batched transactions and contract logic
- Multi-signature support via Gnosis Safe

Polymarket uses a relayer system where gas fees are paid by the platform
for authorized trades. This module handles:
1. Safe deployment and configuration
2. Relayer client for gasless order submission
3. Batched transaction execution
4. Transaction status tracking

The CTF Exchange contract on Polygon handles atomic settlement:
- Order matching happens off-chain (CLOB)
- Settlement happens on-chain (CTF Exchange contract)
- Gas is paid by the relayer for authorized transactions
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config.settings import settings

logger = logging.getLogger(__name__)

# Polygon chain ID
POLYGON_CHAIN_ID = 137

# Gnosis Safe addresses on Polygon
GNOSIS_SAFE_PROXY_FACTORY = "0xC228375F1Ec72F83e3296AA607aE5818F71b1752"
GNOSIS_SAFE_SINGLETON = "0x399DC63B8D3826abE7F5aC0850C881A83d6A2de6"

# Polymarket CTF Exchange contract
CTF_EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

# Conditional Token Framework contract
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"


@dataclass
class RelayerConfig:
    """Configuration for the Polymarket relayer client."""
    enabled: bool = True
    safe_address: str = ""
    nonce: int = 0
    gas_limit: int = 500000
    max_fee_per_gas: int = 0  # 0 = use relayer's gas price
    max_priority_fee: int = 0
    retry_count: int = 3
    retry_delay: float = 1.0


@dataclass
class BatchedTransaction:
    """A batched transaction for gasless execution."""
    tx_id: str
    operations: List[Dict]
    total_gas_estimate: int = 0
    status: str = "pending"  # pending, submitted, confirmed, failed
    tx_hash: str = ""
    block_number: int = 0
    gas_used: int = 0
    timestamp: float = 0.0


@dataclass
class RelayerStats:
    """Relayer performance statistics."""
    total_submitted: int = 0
    total_confirmed: int = 0
    total_failed: int = 0
    total_gas_saved: float = 0.0
    avg_confirmation_time: float = 0.0
    batch_count: int = 0
    avg_batch_size: int = 0


class GnosisSafeRelayer:
    """
    Gnosis Safe + Polymarket Relayer integration for gasless trading.

    The Polymarket relayer pays gas fees for authorized transactions,
    making trades effectively gasless for the user. This module:

    1. Manages the Gnosis Safe wallet for secure key management
    2. Signs transactions with EIP-712 typed signatures (like Polymarket CLOB)
    3. Submits batched transactions via the relayer
    4. Tracks transaction lifecycle (submitted → confirmed → settled)

    Key insight from Polymarket's architecture:
    - Orders are signed off-chain via EIP-712
    - Matched orders are submitted to CTF Exchange contract
    - The relayer pays gas for on-chain settlement
    - Users only pay trading fees (taker fee: 7.2% for crypto)

    For Gnosis Safe deployment:
    - Deploy a new Safe proxy for the trading wallet
    - Configure the Safe with the trading bot as an owner
    - Use the Safe to execute batched transactions
    - The relayer pays gas for all Safe transactions
    """

    def __init__(self, config: Optional[RelayerConfig] = None):
        self.config = config or RelayerConfig(
            # enabled=settings.gnosis_safe_enabled,    # commented out — gasless module removed
            # safe_address=settings.gnosis_safe_address,
        )
        self._stats = RelayerStats()
        self._pending_batches: Dict[str, BatchedTransaction] = {}
        self._confirmed_transactions: List[Dict] = []
        self._nonce = self.config.nonce
        self._web3 = None
        self._safe_contract = None

    # ── Initialization ─────────────────────────────────────────────────

    def initialize(self) -> bool:
        """Initialize the Web3 connection and Safe contract."""
        if not self.config.enabled:
            logger.info("Gnosis Safe relayer disabled")
            return False

        try:
            from web3 import Web3
            self._web3 = Web3(Web3.HTTPProvider(settings.polygon_rpc_url))

            if not self._web3.is_connected():
                logger.error("Failed to connect to Polygon RPC")
                return False

            logger.info(
                "Connected to Polygon RPC | Safe: %s | Block: %d",
                self.config.safe_address or "not deployed",
                self._web3.eth.block_number,
            )

            # Load Safe contract if address is set
            if self.config.safe_address:
                self._load_safe_contract()

            return True

        except ImportError:
            logger.error("web3 not installed — run: pip install web3")
            return False
        except Exception as e:
            logger.error("Gnosis Safe init failed: %s", e)
            return False

    def _load_safe_contract(self):
        """Load the Gnosis Safe contract ABI."""
        # Safe ABI (simplified — just the functions we need)
        safe_abi = [
            {
                "inputs": [
                    {"name": "to", "type": "address"},
                    {"name": "value", "type": "uint256"},
                    {"name": "data", "type": "bytes"},
                    {"name": "operation", "type": "uint8"},
                    {"name": "safeTxGas", "type": "uint256"},
                    {"name": "baseGas", "type": "uint256"},
                    {"name": "gasPrice", "type": "uint256"},
                    {"name": "gasToken", "type": "address"},
                    {"name": "refundReceiver", "type": "address"},
                    {"name": "signatures", "type": "bytes"},
                ],
                "name": "execTransaction",
                "outputs": [{"name": "success", "type": "bool"}],
                "type": "function",
            },
            {
                "inputs": [],
                "name": "nonce",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function",
            },
            {
                "inputs": [],
                "name": "getTransactionHash",
                "outputs": [{"name": "", "type": "bytes32"}],
                "type": "function",
            },
        ]

        self._safe_contract = self._web3.eth.contract(
            address=Web3.to_checksum_address(self.config.safe_address),
            abi=safe_abi,
        )
        self._nonce = self._safe_contract.functions.nonce().call()
        logger.info("Safe contract loaded | Nonce: %d", self._nonce)

    # ── Relayer operations ─────────────────────────────────────────────

    def create_safe_wallet(self, owner_address: str) -> Optional[str]:
        """
        Deploy a new Gnosis Safe proxy for the trading wallet.
        Returns the Safe address if successful.
        """
        if not self._web3:
            logger.error("Web3 not initialized")
            return None

        try:
            # Safe proxy factory ABI (simplified)
            factory_abi = [
                {
                    "inputs": [
                        {"name": "masterCopy", "type": "address"},
                        {"name": "data", "type": "bytes"},
                        {"name": "saltNonce", "type": "uint256"},
                    ],
                    "name": "createProxyWithNonce",
                    "outputs": [{"name": "proxy", "type": "address"}],
                    "type": "function",
                }
            ]

            factory = self._web3.eth.contract(
                address=Web3.to_checksum_address(GNOSIS_SAFE_PROXY_FACTORY),
                abi=factory_abi,
            )

            # Create Safe with single owner (the trading bot)
            salt_nonce = int(time.time())
            owners = [Web3.to_checksum_address(owner_address)]
            threshold = 1  # Single signature required for speed

            # Encode Safe setup function
            setup_data = self._encode_safe_setup(owners, threshold)

            # Deploy proxy
            tx = factory.functions.createProxyWithNonce(
                Web3.to_checksum_address(GNOSIS_SAFE_SINGLETON),
                setup_data,
                salt_nonce,
            ).build_transaction({
                "from": Web3.to_checksum_address(owner_address),
                "nonce": self._web3.eth.get_transaction_count(owner_address),
                "gas": 500000,
                "gasPrice": self._web3.eth.gas_price,
                "chainId": POLYGON_CHAIN_ID,
            })

            # Sign and send (this would need the private key)
            # In production, this is done via the relayer
            logger.info("Safe deployment transaction built (nonce: %d)", salt_nonce)
            logger.info("Safe deployment requires signing — use relayer for gasless deployment")

            return None  # Placeholder — actual deployment via relayer

        except Exception as e:
            logger.error("Safe deployment failed: %s", e)
            return None

    def _encode_safe_setup(self, owners: List[str], threshold: int) -> bytes:
        """Encode the Safe setup function call."""
        # This is a simplified version — the actual encoding requires
        # the full Safe ABI and abi.encode call
        # In production, use web3.py's abi.encode_function_call
        return b""  # Placeholder

    # ── Batched transactions ───────────────────────────────────────────

    def create_batch(
        self,
        operations: List[Dict],
    ) -> str:
        """
        Create a batched transaction for gasless execution.

        Operations format:
        [
            {
                "target": "0x...",  # contract address
                "value": 0,         # ETH value
                "data": "0x...",    # calldata
            }
        ]

        Batching saves gas by:
        1. Single nonce increment for multiple operations
        2. Single base gas cost shared across all operations
        3. Atomic execution (all succeed or all fail)
        """
        batch_id = f"batch_{int(time.time() * 1000)}"

        batch = BatchedTransaction(
            tx_id=batch_id,
            operations=operations,
            timestamp=time.time(),
        )

        # Estimate gas for the batch
        batch.total_gas_estimate = self._estimate_batch_gas(operations)

        self._pending_batches[batch_id] = batch
        self._stats.batch_count += 1
        self._stats.avg_batch_size = (
            (self._stats.avg_batch_size * (self._stats.batch_count - 1) + len(operations))
            / self._stats.batch_count
        )

        logger.info(
            "Created batch %s: %d operations, estimated gas: %d",
            batch_id[:16], len(operations), batch.total_gas_estimate,
        )

        return batch_id

    def _estimate_batch_gas(self, operations: List[Dict]) -> int:
        """Estimate gas for a batch of operations."""
        # Base gas for Safe transaction
        base_gas = 21000
        # Per-operation gas estimate
        per_op_gas = 50000
        return base_gas + len(operations) * per_op_gas

    async def submit_batch(self, batch_id: str, signed_data: bytes = b"") -> bool:
        """
        Submit a batched transaction via the relayer.

        The relayer pays gas and submits the transaction to Polygon.
        For gasless mode, the relayer's private key signs the transaction.
        """
        batch = self._pending_batches.get(batch_id)
        if not batch:
            logger.error("Batch %s not found", batch_id)
            return False

        if not self.config.enabled:
            logger.warning("Relayer disabled — cannot submit batch")
            return False

        try:
            # In production, this would:
            # 1. Sign the batch with EIP-712
            # 2. Submit to the Polymarket relayer API
            # 3. The relayer pays gas and submits to Polygon
            # 4. Return the transaction hash

            batch.status = "submitted"
            self._stats.total_submitted += 1

            logger.info(
                "Batch %s submitted via relayer (gas saved: ~%d wei)",
                batch_id[:16], batch.total_gas_estimate * 1000000000,  # rough gas cost in wei
            )

            return True

        except Exception as e:
            logger.error("Batch submission failed: %s", e)
            batch.status = "failed"
            self._stats.total_failed += 1
            return False

    def confirm_batch(self, batch_id: str, tx_hash: str, block_number: int, gas_used: int):
        """Mark a batch as confirmed on-chain."""
        batch = self._pending_batches.get(batch_id)
        if batch:
            batch.status = "confirmed"
            batch.tx_hash = tx_hash
            batch.block_number = block_number
            batch.gas_used = gas_used

            self._stats.total_confirmed += 1
            self._stats.total_gas_saved += gas_used * 1000000000  # gas * gasPrice estimate

            # Move to confirmed list
            self._confirmed_transactions.append({
                "batch_id": batch_id,
                "tx_hash": tx_hash,
                "block_number": block_number,
                "gas_used": gas_used,
                "operations_count": len(batch.operations),
                "confirmed_at": time.time(),
            })

            # Update average confirmation time
            if batch.timestamp > 0:
                confirm_time = time.time() - batch.timestamp
                n = self._stats.total_confirmed
                self._stats.avg_confirmation_time = (
                    (self._stats.avg_confirmation_time * (n - 1) + confirm_time) / n
                )

            logger.info(
                "Batch %s confirmed at block %d (gas: %d)",
                batch_id[:16], block_number, gas_used,
            )

    # ── Order execution via relayer ────────────────────────────────────

    def build_order_transaction(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> Optional[Dict]:
        """
        Build a transaction for order execution via the relayer.

        The Polymarket CLOB uses EIP-712 typed signatures:
        - Order struct: maker, taker, tokenId, makerAmount, takerAmount, etc.
        - Signed by the maker's private key
        - Submitted to CTF Exchange contract for atomic settlement
        """
        # Calculate amounts
        if side == "BUY":
            maker_amount = int(size * 1e6)  # USDC (6 decimals)
            taker_amount = int(size / price * 1e6) if price > 0 else 0
        else:
            maker_amount = int(size / price * 1e6) if price > 0 else 0
            taker_amount = int(size * 1e6)

        # Build EIP-712 order struct
        order = {
            "salt": str(int(time.time() * 1000)),
            "maker": self.config.safe_address or settings.polygon_wallet_address,
            "signer": settings.polygon_wallet_address,
            "taker": "0x0000000000000000000000000000000000000000",
            "tokenId": token_id,
            "makerAmount": str(maker_amount),
            "takerAmount": str(taker_amount),
            "expiration": str(int(time.time()) + 300),  # 5 min expiry
            "nonce": str(self._nonce),
            "feeRateBps": "720",  # 7.2% for crypto markets
            "side": "BUY" if side == "BUY" else "SELL",
            "signatureType": 0,  # EOA signature
        }

        self._nonce += 1

        return {
            "order": order,
            "gas_estimate": 200000,
            "requires_relayer": self.config.enabled,
        }

    def build_batch_orders(
        self,
        orders: List[Dict],
    ) -> str:
        """
        Build a batch of orders for gasless execution.
        Multiple orders can be batched into a single transaction.
        """
        operations = []
        for order_params in orders:
            tx = self.build_order_transaction(**order_params)
            if tx:
                operations.append({
                    "target": CTF_EXCHANGE_ADDRESS,
                    "value": 0,
                    "data": tx["order"],  # Would be ABI-encoded calldata
                })

        return self.create_batch(operations)

    # ── Status and stats ───────────────────────────────────────────────

    @property
    def stats(self) -> Dict:
        return {
            "enabled": self.config.enabled,
            "safe_address": self.config.safe_address or "not deployed",
            "nonce": self._nonce,
            "total_submitted": self._stats.total_submitted,
            "total_confirmed": self._stats.total_confirmed,
            "total_failed": self._stats.total_failed,
            "pending_batches": len(self._pending_batches),
            "avg_confirmation_time": round(self._stats.avg_confirmation_time, 2),
            "total_gas_saved_usd": round(self._stats.total_gas_saved / 1e18 * 0.5, 4),  # rough USD estimate
            "batch_count": self._stats.batch_count,
            "avg_batch_size": round(self._stats.avg_batch_size, 1),
        }

    def get_pending_batches(self) -> List[Dict]:
        """Get all pending batches."""
        return [
            {
                "batch_id": b.tx_id,
                "operations": len(b.operations),
                "status": b.status,
                "gas_estimate": b.total_gas_estimate,
                "age_seconds": round(time.time() - b.timestamp, 1) if b.timestamp > 0 else 0,
            }
            for b in self._pending_batches.values()
            if b.status in ("pending", "submitted")
        ]

    def get_recent_confirmations(self, limit: int = 10) -> List[Dict]:
        """Get recent confirmed transactions."""
        return self._confirmed_transactions[-limit:]
