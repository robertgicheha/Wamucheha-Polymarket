"""
Read-only monitoring dashboard with manual withdrawal endpoint.
Displays compounding metrics, trade history, and allows manual profit withdrawal.

Run standalone:  python -m dashboard.app
In production, main.py starts this in a background thread.
"""
from flask import Flask, jsonify, render_template_string, request

from config.settings import settings
from dashboard.state import get_state

app = Flask(__name__)

PAGE = """
<!doctype html>
<html>
<head>
  <title>Polymarket Bot Dashboard</title>
  <meta http-equiv="refresh" content="15">
  <style>
    body { font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 2rem; }
    h1 { color: #58a6ff; }
    h2 { color: #8b949e; margin-top: 1.5rem; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
            padding: 1rem; margin-bottom: 1rem; }
    .metric { display: inline-block; margin-right: 2rem; min-width: 120px; }
    .metric .label { color: #8b949e; font-size: 0.85rem; }
    .metric .value { font-size: 1.4rem; }
    .metric .value.profit { color: #3fb950; }
    .metric .value.loss { color: #f85149; }
    .halted { color: #f85149; font-weight: bold; }
    .ok { color: #3fb950; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 0.4rem; border-bottom: 1px solid #30363d; }
    .withdraw-form { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                     padding: 1.5rem; margin-top: 1rem; }
    .withdraw-form input { background: #0d1117; color: #c9d1d9; border: 1px solid #30363d;
                           padding: 0.5rem; border-radius: 4px; width: 200px; margin-right: 0.5rem; }
    .withdraw-form button { background: #238636; color: white; border: none; padding: 0.5rem 1rem;
                            border-radius: 4px; cursor: pointer; }
    .withdraw-form button:hover { background: #2ea043; }
    .withdraw-form button:disabled { background: #30363d; cursor: not-allowed; }
    .info { color: #8b949e; font-size: 0.9rem; margin-top: 0.5rem; }
    nav { margin-bottom: 2rem; padding: 0.8rem 0; border-bottom: 1px solid #30363d; }
    nav a { color: #58a6ff; text-decoration: none; margin-right: 1.5rem; font-size: 1rem;
            padding: 0.3rem 0.8rem; border-radius: 4px; }
    nav a:hover { background: #161b22; }
    nav a.active { background: #238636; color: white; }
    .training-badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
                      font-size: 0.8rem; margin-left: 0.5rem; }
    .training-on { background: #238636; color: white; }
    .training-off { background: #30363d; color: #8b949e; }
  </style>
</head>
<body>
  <nav>
    <a href="/" class="active">Dashboard</a>
    <a href="/paper">Paper Trade (Training)
      {% if state.training_mode %}
      <span class="training-badge training-on">ON</span>
      {% else %}
      <span class="training-badge training-off">OFF</span>
      {% endif %}
    </a>
  </nav>

  <h1>Polymarket Bot -- {{ state.mode }} mode
    {% if state.training_mode %}
    <span class="training-badge training-on">TRAINING</span>
    {% else %}
    <span class="training-badge training-off">LIVE READY</span>
    {% endif %}
  </h1>

  <div class="card">
    <h2>Compounding Status</h2>
    <div class="metric"><div class="label">Bankroll</div><div class="value">${{ "%.2f"|format(state.bankroll) }}</div></div>
    <div class="metric"><div class="label">Principal</div><div class="value">${{ "%.2f"|format(state.principal) }}</div></div>
    <div class="metric"><div class="label">Profit</div>
      <div class="value {{ 'profit' if state.profit >= 0 else 'loss' }}">
        ${{ "%.2f"|format(state.profit) }} ({{ "%.1f"|format(state.profit_pct) }}%)
      </div>
    </div>
    <div class="metric"><div class="label">Peak Bankroll</div><div class="value">${{ "%.2f"|format(state.peak_bankroll) }}</div></div>
    <div class="metric"><div class="label">Drawdown</div>
      <div class="value {{ 'loss' if state.drawdown_pct > 0 else 'ok' }}">{{ "%.1f"|format(state.drawdown_pct) }}%</div>
    </div>
    <div class="metric"><div class="label">Growth Rate (ann.)</div>
      <div class="value">{{ "%.1f"|format(state.compound_growth_rate) }}%</div>
    </div>
    <div class="metric"><div class="label">Total Withdrawn</div><div class="value">${{ "%.2f"|format(state.total_withdrawn) }}</div></div>
  </div>

  <div class="card">
    <h2>Trade Stats</h2>
    <div class="metric"><div class="label">Total Trades</div><div class="value">{{ state.total_trades }}</div></div>
    <div class="metric"><div class="label">Wins</div><div class="value ok">{{ state.total_wins }}</div></div>
    <div class="metric"><div class="label">Losses</div><div class="value loss">{{ state.total_losses }}</div></div>
    <div class="metric"><div class="label">Win Rate</div><div class="value">{{ "%.1f"|format(state.win_rate) }}%</div></div>
    <div class="metric"><div class="label">Consec. Losses</div><div class="value">{{ state.consecutive_losses }}</div></div>
    <div class="metric"><div class="label">Status</div>
      <div class="value {{ 'halted' if state.halted else 'ok' }}">
        {{ 'HALTED: ' + state.halt_reason if state.halted else 'RUNNING' }}
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Manual Withdrawal</h2>
    <form class="withdraw-form" id="withdrawForm" onsubmit="submitWithdraw(event)">
      <label>Amount (USD):</label>
      <input type="number" id="withdrawAmount" step="0.01" min="0.01"
             max="{{ state.profit }}" placeholder="0.00">
      <button type="submit" {{ 'disabled' if state.profit <= 0 }}>Withdraw</button>
      <div class="info">
        Available to withdraw: ${{ "%.2f"|format(state.profit) }} |
        Bot will send USDC to your configured destination address.
      </div>
    </form>
    <div id="withdrawResult" style="margin-top: 0.5rem;"></div>
  </div>

  {% if state.withdrawal_history %}
  <div class="card">
    <h2>Withdrawal History</h2>
    <table>
      <tr><th>Amount</th><th>Destination</th><th>Status</th><th>Time</th></tr>
      {% for w in state.withdrawal_history %}
      <tr>
        <td>${{ "%.2f"|format(w.amount_usd) }}</td>
        <td>{{ w.destination[:20] }}{% if w.destination|length > 20 %}...{% endif %}</td>
        <td class="{{ 'ok' if w.status == 'completed' else 'loss' }}">{{ w.status }}</td>
        <td>{{ w.timestamp }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>
  {% endif %}

  <div class="card">
    <h2>Open Positions ({{ state.open_positions|length }})</h2>
    <table>
      <tr><th>Market</th><th>Category</th><th>Side</th><th>Entry</th><th>Size</th><th>Stop-loss</th></tr>
      {% for p in state.open_positions %}
      <tr><td>{{ p.market_id }}</td><td>{{ p.category }}</td><td>{{ p.side }}</td>
          <td>{{ "%.3f"|format(p.entry_price) }}</td><td>${{ "%.2f"|format(p.size_usd) }}</td>
          <td>{{ "%.3f"|format(p.stop_loss_price) }}</td></tr>
      {% endfor %}
    </table>
  </div>

  <div class="card">
    <h2>Recent Trades</h2>
    <table>
      <tr><th>Market</th><th>Category</th><th>P&amp;L</th><th>Closed</th></tr>
      {% for t in state.recent_trades %}
      <tr><td>{{ t.market_id }}</td><td>{{ t.category }}</td>
          <td class="{{ 'ok' if t.pnl_usd >= 0 else 'loss' }}">${{ "%.2f"|format(t.pnl_usd) }}</td>
          <td>{{ t.closed_at }}</td></tr>
      {% endfor %}
    </table>
  </div>

  <script>
  function submitWithdraw(e) {
    e.preventDefault();
    var amount = document.getElementById('withdrawAmount').value;
    if (!amount || parseFloat(amount) <= 0) return;
    fetch('/api/withdraw', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({amount: parseFloat(amount)})
    })
    .then(r => r.json())
    .then(data => {
      document.getElementById('withdrawResult').innerHTML =
        '<span style="color:' + (data.success ? '#3fb950' : '#f85149') + '">' +
        data.message + '</span>';
    })
    .catch(err => {
      document.getElementById('withdrawResult').innerHTML =
        '<span style="color:#f85149">Error: ' + err + '</span>';
    });
  }
  </script>
</body>
</html>
"""


PAPER_PAGE = """
<!doctype html>
<html>
<head>
  <title>Paper Trading - Training Mode</title>
  <meta http-equiv="refresh" content="15">
  <style>
    body { font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 2rem; }
    h1 { color: #58a6ff; }
    h2 { color: #8b949e; margin-top: 1.5rem; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
            padding: 1rem; margin-bottom: 1rem; }
    .metric { display: inline-block; margin-right: 2rem; min-width: 120px; }
    .metric .label { color: #8b949e; font-size: 0.85rem; }
    .metric .value { font-size: 1.4rem; }
    .metric .value.profit { color: #3fb950; }
    .metric .value.loss { color: #f85149; }
    .ok { color: #3fb950; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 0.4rem; border-bottom: 1px solid #30363d; }
    nav { margin-bottom: 2rem; padding: 0.8rem 0; border-bottom: 1px solid #30363d; }
    nav a { color: #58a6ff; text-decoration: none; margin-right: 1.5rem; font-size: 1rem;
            padding: 0.3rem 0.8rem; border-radius: 4px; }
    nav a:hover { background: #161b22; }
    nav a.active { background: #238636; color: white; }
    .training-badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
                      font-size: 0.8rem; margin-left: 0.5rem; }
    .training-on { background: #238636; color: white; }
    .training-off { background: #30363d; color: #8b949e; }
    .disabled-overlay { opacity: 0.4; pointer-events: none; }
    .leaderboard-rank { font-size: 1.2rem; }
    .rank-1 { color: #ffd700; }
    .rank-2 { color: #c0c0c0; }
    .rank-3 { color: #cd7f32; }
    .analytics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; }
  </style>
</head>
<body>
  <nav>
    <a href="/">Dashboard</a>
    <a href="/paper" class="active">Paper Trade (Training)
      {% if state.training_mode %}
      <span class="training-badge training-on">ON</span>
      {% else %}
      <span class="training-badge training-off">OFF</span>
      {% endif %}
    </a>
  </nav>

  <h1>Paper Trading (Training Mode)
    {% if state.training_mode %}
    <span class="training-badge training-on">ACTIVE</span>
    {% else %}
    <span class="training-badge training-off">DISABLED</span>
    {% endif %}
  </h1>

  {% if not state.training_mode %}
  <div class="card" style="border-color: #f85149;">
    <h2 style="color: #f85149;">Training Mode Disabled</h2>
    <p>Set <code>TRAINING_MODE=true</code> in config/.env to enable paper trading.</p>
    <p>The bot is configured for live trading. No paper orders will be executed.</p>
  </div>
  {% endif %}

  <div class="card">
    <h2>Account Balance</h2>
    <div class="metric"><div class="label">Cash</div><div class="value">${{ "%.2f"|format(state.paper_balance) }}</div></div>
    <div class="metric"><div class="label">Starting Balance</div><div class="value">${{ "%.2f"|format(state.paper_starting_balance) }}</div></div>
    <div class="metric"><div class="label">Positions Value</div><div class="value">${{ "%.2f"|format(state.paper_positions_value) }}</div></div>
    <div class="metric"><div class="label">Total Value</div><div class="value">${{ "%.2f"|format(state.paper_total_value) }}</div></div>
    <div class="metric"><div class="label">P&amp;L</div>
      <div class="value {{ 'profit' if state.paper_pnl >= 0 else 'loss' }}">
        ${{ "%.2f"|format(state.paper_pnl) }}
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Performance Analytics</h2>
    {% if state.paper_analytics %}
    <div class="analytics-grid">
      <div class="metric"><div class="label">ROI</div>
        <div class="value {{ 'profit' if state.paper_analytics.get('roi_pct', 0) >= 0 else 'loss' }}">
          {{ "%.1f"|format(state.paper_analytics.get('roi_pct', 0)) }}%
        </div>
      </div>
      <div class="metric"><div class="label">Total Trades</div><div class="value">{{ state.paper_analytics.get('total_trades', 0) }}</div></div>
      <div class="metric"><div class="label">Buy Count</div><div class="value">{{ state.paper_analytics.get('buy_count', 0) }}</div></div>
      <div class="metric"><div class="label">Sell Count</div><div class="value">{{ state.paper_analytics.get('sell_count', 0) }}</div></div>
      <div class="metric"><div class="label">Win Rate</div>
        <div class="value {{ 'ok' if state.paper_analytics.get('win_rate', 0) > 0.5 else '' }}">
          {{ "%.1f"|format(state.paper_analytics.get('win_rate', 0) * 100) }}%
        </div>
      </div>
      <div class="metric"><div class="label">Sharpe Ratio</div>
        <div class="value">{{ "%.2f"|format(state.paper_analytics.get('sharpe_ratio', 0)) }}</div>
      </div>
      <div class="metric"><div class="label">Max Drawdown</div>
        <div class="value loss">{{ "%.1f"|format(state.paper_analytics.get('max_drawdown', 0) * 100) }}%</div>
      </div>
      <div class="metric"><div class="label">Total Fees</div>
        <div class="value">${{ "%.4f"|format(state.paper_analytics.get('total_fees', 0)) }}</div>
      </div>
      <div class="metric"><div class="label">Avg Trade Size</div>
        <div class="value">${{ "%.2f"|format(state.paper_analytics.get('avg_trade_size', 0)) }}</div>
      </div>
    </div>
    {% else %}
    <p style="color: #8b949e;">No analytics yet. Start trading to see performance metrics.</p>
    {% endif %}
  </div>

  <div class="card">
    <h2>Open Positions ({{ state.paper_positions|length }})</h2>
    {% if state.paper_positions %}
    <table>
      <tr><th>Market</th><th>Shares</th><th>Cost Basis</th><th>Avg Entry</th></tr>
      {% for slug, pos in state.paper_positions.items() %}
      <tr>
        <td>{{ slug }}</td>
        <td>{{ "%.4f"|format(pos.get('shares', 0)) }}</td>
        <td>${{ "%.4f"|format(pos.get('cost_basis', 0)) }}</td>
        <td>{{ "%.4f"|format(pos.get('avg_entry_price', pos.get('cost_basis', 0) / pos.get('shares', 1))) }}</td>
      </tr>
      {% endfor %}
    </table>
    {% else %}
    <p style="color: #8b949e;">No open positions.</p>
    {% endif %}
  </div>

  <div class="card">
    <h2>Trade Stats</h2>
    <div class="metric"><div class="label">Total Trades</div><div class="value">{{ state.paper_total_trades }}</div></div>
    <div class="metric"><div class="label">Total Fees</div><div class="value">${{ "%.4f"|format(state.paper_total_fees) }}</div></div>
    <div class="metric"><div class="label">Backend</div><div class="value" style="font-size: 1rem;">{{ state.paper_backend }}</div></div>
  </div>

  <div class="card">
    <h2>CLI Quick Reference</h2>
    <table>
      <tr><th>Command</th><th>Description</th></tr>
      <tr><td><code>pm-trader init --balance 10000</code></td><td>Create paper trading account</td></tr>
      <tr><td><code>pm-trader markets search "bitcoin"</code></td><td>Search markets</td></tr>
      <tr><td><code>pm-trader buy SLUG yes 500</code></td><td>Buy $500 of YES shares</td></tr>
      <tr><td><code>pm-trader sell SLUG yes 100</code></td><td>sell 100 shares</td></tr>
      <tr><td><code>pm-trader portfolio</code></td><td>Show open positions with live P&amp;L</td></tr>
      <tr><td><code>pm-trader stats</code></td><td>Performance analytics</td></tr>
      <tr><td><code>pm-trader stats --card</code></td><td>Generate shareable stats card</td></tr>
      <tr><td><code>pm-trader history</code></td><td>Trade history</td></tr>
      <tr><td><code>pm-trader leaderboard</code></td><td>Local account rankings</td></tr>
    </table>
  </div>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE, state=get_state())


@app.route("/paper")
def paper_page():
    return render_template_string(PAPER_PAGE, state=get_state())


@app.route("/api/state")
def api_state():
    s = get_state()
    return jsonify({
        "mode": s.mode,
        "training_mode": s.training_mode,
        "bankroll": s.bankroll,
        "principal": s.principal,
        "profit": s.profit,
        "profit_pct": s.profit_pct,
        "total_withdrawn": s.total_withdrawn,
        "peak_bankroll": s.peak_bankroll,
        "compound_growth_rate": s.compound_growth_rate,
        "drawdown_pct": s.drawdown_pct,
        "win_rate": s.win_rate,
        "total_trades": s.total_trades,
        "halted": s.halted,
        "halt_reason": s.halt_reason,
        "open_positions": len(s.open_positions),
        "withdrawal_history": [
            {"amount": w.amount_usd, "destination": w.destination, "status": w.status, "time": str(w.timestamp)}
            for w in s.withdrawal_history
        ],
        "paper": {
            "balance": s.paper_balance,
            "starting_balance": s.paper_starting_balance,
            "positions_value": s.paper_positions_value,
            "total_value": s.paper_total_value,
            "pnl": s.paper_pnl,
            "positions": s.paper_positions,
            "total_trades": s.paper_total_trades,
            "total_fees": s.paper_total_fees,
            "analytics": s.paper_analytics,
        },
    })


@app.route("/api/withdraw", methods=["POST"])
def api_withdraw():
    """Manual withdrawal endpoint. Triggers USDC transfer from the trading wallet."""
    data = request.get_json()
    amount = data.get("amount", 0)
    destination = data.get("destination", "")

    if amount <= 0:
        return jsonify({"success": False, "message": "Amount must be > $0"})

    # Import here to avoid circular imports
    from risk.risk_manager import RiskManager
    from alerts.notifier import Severity, notifier

    # We need to signal to the main loop to execute the withdrawal
    # Store the request in a shared state that main.py picks up
    from dashboard.state import get_state
    state = get_state()

    if not settings.withdrawal_enabled:
        return jsonify({"success": False, "message": "Withdrawals are disabled in config"})

    if amount > state.profit:
        return jsonify({
            "success": False,
            "message": f"Insufficient profit. Available: ${state.profit:.2f}",
        })

    # Store withdrawal request for main.py to process
    _pending_withdrawal = {"amount": amount, "destination": destination}
    app.config["PENDING_WITHDRAWAL"] = _pending_withdrawal

    notifier.send(
        f"Withdrawal requested: ${amount:.2f} USDC — processing...",
        Severity.INFO,
    )

    return jsonify({
        "success": True,
        "message": f"Withdrawal of ${amount:.2f} requested. Processing on next cycle.",
    })


def get_pending_withdrawal():
    """Called by main.py each cycle to check for pending withdrawal requests."""
    return app.config.pop("PENDING_WITHDRAWAL", None)


def run():
    app.run(host=settings.dashboard_host, port=settings.dashboard_port)


if __name__ == "__main__":
    run()
