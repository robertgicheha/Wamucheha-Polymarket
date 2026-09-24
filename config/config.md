1. Create a fresh Polymarket wallet — sign up at polymarket.com with a wallet that has never been used anywhere else (no existing MetaMask wallet, nothing that's touched any repo or chat). This becomes your funder/signer.

2. Get a Polygon RPC URL — free tier at dashboard.alchemy.com or infura.io, select "Polygon Mainnet."

3. Fund it — send USDC on Polygon to your Polymarket deposit address (shown in your polymarket.com profile → Deposit). Given LIVE_MAX_ORDER_USD=$10 / LIVE_MAX_OPEN_EXPOSURE_USD=$30 are already capped in the code, even $30–50 is enough to start.

4. Put the private key and RPC URL directly on the server yourself — SSH in (ssh -L 8080:localhost:8080 botadmin@161.97.163.223), then edit the env file directly (sudo -u botsvc nano /opt/bots/wamucheha-poly/.env) and add:


POLYMARKET_PRIVATE_KEY=0x...
POLYGON_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY
LIVE_TRADING_ACK=I understand this trades real money
Don't paste the key to me — I never need to see it, and it shouldn't sit in a chat transcript.

5. Tell me when that's done. I'll run scripts/preflight.py against it (checks approvals, balance, gas, that the key never leaked into git history), fix anything it flags, and once it says READY we set ALLOW_BASELINE_LIVE=true (or wait for more paper trades — your call) and flip TRADING_MODE=live with a restart.