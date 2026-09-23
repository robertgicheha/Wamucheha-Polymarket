"""
Read-only monitoring dashboard with manual withdrawal endpoint.
Displays compounding metrics, real-time BTC/lifecycle-engine state, model
accuracy, trade history, arbitrage stats, and a log tail.

Run standalone:  python -m dashboard.app
In production, main.py starts this in a background thread.

Auth: if DASHBOARD_USERNAME/DASHBOARD_PASSWORD are set in .env, every route
requires HTTP Basic Auth. If they're NOT set, the dashboard is wide open —
fine for localhost-only access, dangerous if the port is reachable from the
public internet (the /api/withdraw endpoint moves money). Set them before
exposing this beyond an SSH tunnel / localhost.
"""
import os
import time
from functools import wraps

from flask import Flask, Response, jsonify, render_template_string, request

from config.settings import settings
from dashboard.state import get_state

app = Flask(__name__)

BOT_LOG_PATH = os.environ.get("BOT_LOG_PATH", "logs/bot.log")


# ── Auth ─────────────────────────────────────────────────────────────────

def _auth_configured() -> bool:
    return bool(settings.dashboard_username and settings.dashboard_password)


def _check_auth(auth) -> bool:
    return (
        auth is not None
        and auth.username == settings.dashboard_username
        and auth.password == settings.dashboard_password
    )


def require_auth(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not _auth_configured():
            return view(*args, **kwargs)
        if not _check_auth(request.authorization):
            return Response(
                "Authentication required", 401,
                {"WWW-Authenticate": 'Basic realm="Polymarket Bot Dashboard"'},
            )
        return view(*args, **kwargs)
    return wrapped


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
    .metric { display: inline-block; margin-right: 2rem; min-width: 120px; margin-bottom: 0.5rem; }
    .metric .label { color: #8b949e; font-size: 0.85rem; }
    .metric .value { font-size: 1.4rem; }
    .metric .value.small { font-size: 1.1rem; }
    .metric .value.profit { color: #3fb950; }
    .metric .value.loss { color: #f85149; }
    .halted { color: #f85149; font-weight: bold; }
    .ok { color: #3fb950; }
    .warn { color: #d29922; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 0.4rem; border-bottom: 1px solid #30363d; font-size: 0.9rem; }
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
    .badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
             font-size: 0.8rem; margin-left: 0.5rem; }
    .badge-on { background: #238636; color: white; }
    .badge-off { background: #30363d; color: #8b949e; }
    .badge-warn { background: #9e6a03; color: white; }
    .security-banner { background: #3d1e00; border: 1px solid #9e6a03; color: #ffcc80;
                        padding: 0.8rem 1rem; border-radius: 8px; margin-bottom: 1rem; }
  </style>
</head>
<body>
  <nav>
    <a href="/" class="active">Dashboard</a>
    <a href="/paper">Paper Trade (Training)
      {% if state.training_mode %}<span class="badge badge-on">ON</span>
      {% else %}<span class="badge badge-off">OFF</span>{% endif %}
    </a>
    <a href="/logs">Logs</a>
  </nav>

  {% if not auth_configured %}
  <div class="security-banner">
    <strong>No dashboard password set.</strong> This page (and the withdraw
    endpoint) is open to anyone who can reach this port. Set
    DASHBOARD_USERNAME / DASHBOARD_PASSWORD in .env before exposing this
    beyond localhost or an SSH tunnel.
  </div>
  {% endif %}

  <h1>Polymarket Bot -- {{ state.mode }} mode
    {% if state.training_mode %}<span class="badge badge-on">TRAINING</span>
    {% else %}<span class="badge badge-off">LIVE READY</span>{% endif %}
    {% if state.halted %}<span class="badge badge-warn">HALTED</span>{% endif %}
  </h1>

  <div class="card">
    <h2>Bot Health</h2>
    <div class="metric"><div class="label">Uptime</div><div class="value small">{{ "%.1f"|format(state.uptime_seconds / 3600) }}h</div></div>
    <div class="metric"><div class="label">Status</div>
      <div class="value small {{ 'halted' if state.halted else 'ok' }}">
        {{ 'HALTED: ' + state.halt_reason if state.halted else 'RUNNING' }}
      </div>
    </div>
    <div class="metric"><div class="label">Strategy</div><div class="value small">{{ state.active_strategy }}</div></div>
    <div class="metric"><div class="label">ML Prediction</div>
      <div class="value small {{ 'ok' if state.ml_prediction_enabled else '' }}">
        {{ 'ENABLED' if state.ml_prediction_enabled else 'fallback heuristic' }}
      </div>
    </div>
    <div class="metric"><div class="label">TTE Models Trained</div>
      <div class="value small">{{ state.tte_models_fitted }} / {{ state.tte_total_models }}</div>
    </div>
    <div class="metric"><div class="label">Last Retrain</div><div class="value small">{{ state.last_retrain or "never" }}</div></div>
  </div>

  <div class="card">
    <h2>Live BTC Price (BRTI)</h2>
    <div class="metric"><div class="label">Price</div><div class="value">${{ "%.2f"|format(state.brti_price) }}</div></div>
    <div class="metric"><div class="label">Spread</div><div class="value small">{{ "%.2f"|format(state.brti_spread_bps) }} bps</div></div>
    <div class="metric"><div class="label">Exchanges</div><div class="value small">{{ state.brti_exchanges_used }}</div></div>
    <div class="metric"><div class="label">Ticks Received</div><div class="value small">{{ state.brti_tick_count }}</div></div>
  </div>

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
  </div>

  <div class="card">
    <h2>Model Accuracy (live)</h2>
    {% if state.model_accuracy.get('n_predictions', 0) > 0 %}
    <div class="metric"><div class="label">Predictions Scored</div><div class="value small">{{ state.model_accuracy.n_predictions }}</div></div>
    <div class="metric"><div class="label">Brier Score</div>
      <div class="value small {{ 'ok' if state.model_accuracy.get('better_than_baseline') else 'warn' }}">
        {{ "%.4f"|format(state.model_accuracy.brier_score) }}
      </div>
    </div>
    <div class="metric"><div class="label">vs. Baseline (0.5)</div><div class="value small">{{ "%.4f"|format(state.model_accuracy.baseline_brier_score) }}</div></div>
    <div class="metric"><div class="label">Beating Baseline?</div>
      <div class="value small {{ 'ok' if state.model_accuracy.get('better_than_baseline') else 'loss' }}">
        {{ 'YES' if state.model_accuracy.get('better_than_baseline') else 'NO' }}
      </div>
    </div>
    <div class="info">Lower Brier is better. Computed from real trade outcomes vs. the model_prob logged at entry — not a backtest number.</div>
    {% else %}
    <p style="color: #8b949e;">No scored predictions yet — needs ML_PREDICTION_ENABLED=true and at least one closed trade with a logged model_prob.</p>
    {% endif %}
  </div>

  <div class="card">
    <h2>5-Minute Lifecycle Engine ({{ state.lifecycle_active_windows|length }} active)</h2>
    <div class="metric"><div class="label">Total P&amp;L</div>
      <div class="value small {{ 'profit' if state.lifecycle_total_pnl >= 0 else 'loss' }}">${{ "%.4f"|format(state.lifecycle_total_pnl) }}</div>
    </div>
    <div class="metric"><div class="label">Win Rate</div><div class="value small">{{ "%.1f"|format(state.lifecycle_win_rate) }}%</div></div>
    <div class="metric"><div class="label">Markets Traded</div><div class="value small">{{ state.lifecycle_total_trades }}</div></div>
    {% if state.lifecycle_active_windows %}
    <table>
      <tr><th>Asset</th><th>Price to Beat</th><th>Current</th><th>Time Left</th><th>Position</th><th>Size</th><th>PnL</th></tr>
      {% for w in state.lifecycle_active_windows %}
      <tr>
        <td>{{ w.asset|upper }}</td>
        <td>${{ "%.2f"|format(w.price_to_beat) }}</td>
        <td>{{ "%.3f"|format(w.current_yes_price) }}</td>
        <td>{{ "%.0f"|format(w.time_remaining) }}s</td>
        <td>{{ w.position_side or "-" }}</td>
        <td>${{ "%.2f"|format(w.position_size_usd) }}</td>
        <td class="{{ 'ok' if w.pnl_usd >= 0 else 'loss' }}">${{ "%.4f"|format(w.pnl_usd) }}</td>
      </tr>
      {% endfor %}
    </table>
    {% else %}
    <p style="color: #8b949e;">No active 5-minute market windows right now.</p>
    {% endif %}
  </div>

  <div class="card">
    <h2>Arbitrage</h2>
    <div class="metric"><div class="label">P&amp;L</div>
      <div class="value small {{ 'profit' if state.arb_pnl >= 0 else 'loss' }}">${{ "%.4f"|format(state.arb_pnl) }}</div>
    </div>
    <div class="metric"><div class="label">Win Rate</div><div class="value small">{{ "%.1f"|format(state.arb_win_rate) }}%</div></div>
    <div class="metric"><div class="label">Open</div><div class="value small">{{ state.arb_open_positions }}</div></div>
    <div class="metric"><div class="label">Closed</div><div class="value small">{{ state.arb_closed_positions }}</div></div>
    <div class="metric"><div class="label">Opportunities Seen</div><div class="value small">{{ state.arb_opportunities }}</div></div>
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
    <h2>Trade Log (persistent, all sources)</h2>
    {% if state.trade_log_recent %}
    <table>
      <tr><th>Time</th><th>Asset</th><th>Side</th><th>Entry</th><th>Exit</th><th>PnL</th><th>Strategy</th><th>Won</th></tr>
      {% for t in state.trade_log_recent %}
      <tr>
        <td>{{ t.timestamp[:19] if t.timestamp else "" }}</td>
        <td>{{ (t.asset or "")|upper }}</td>
        <td>{{ t.entry_side }}</td>
        <td>{{ "%.3f"|format(t.entry_price) if t.entry_price is not none else "-" }}</td>
        <td>{{ "%.3f"|format(t.exit_price) if t.exit_price is not none else "open" }}</td>
        <td class="{{ 'ok' if (t.pnl_usd or 0) >= 0 else 'loss' }}">${{ "%.4f"|format(t.pnl_usd or 0) }}</td>
        <td>{{ t.strategy }}</td>
        <td>{{ "WIN" if t.won else ("LOSS" if t.exit_price is not none else "-") }}</td>
      </tr>
      {% endfor %}
    </table>
    {% else %}
    <p style="color: #8b949e;">No trades logged yet.</p>
    {% endif %}
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
    .badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
             font-size: 0.8rem; margin-left: 0.5rem; }
    .badge-on { background: #238636; color: white; }
    .badge-off { background: #30363d; color: #8b949e; }
    .analytics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; }
  </style>
</head>
<body>
  <nav>
    <a href="/">Dashboard</a>
    <a href="/paper" class="active">Paper Trade (Training)
      {% if state.training_mode %}<span class="badge badge-on">ON</span>
      {% else %}<span class="badge badge-off">OFF</span>{% endif %}
    </a>
    <a href="/logs">Logs</a>
  </nav>

  <h1>Paper Trading (Training Mode)
    {% if state.training_mode %}<span class="badge badge-on">ACTIVE</span>
    {% else %}<span class="badge badge-off">DISABLED</span>{% endif %}
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


LOGS_PAGE = """
<!doctype html>
<html>
<head>
  <title>Bot Logs</title>
  <meta http-equiv="refresh" content="10">
  <style>
    body { font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 2rem; }
    h1 { color: #58a6ff; }
    nav { margin-bottom: 2rem; padding: 0.8rem 0; border-bottom: 1px solid #30363d; }
    nav a { color: #58a6ff; text-decoration: none; margin-right: 1.5rem; font-size: 1rem;
            padding: 0.3rem 0.8rem; border-radius: 4px; }
    nav a:hover { background: #161b22; }
    nav a.active { background: #238636; color: white; }
    pre { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem;
          overflow-x: auto; font-size: 0.85rem; line-height: 1.4; white-space: pre-wrap; word-break: break-all; }
    .line-error { color: #f85149; }
    .line-warn { color: #d29922; }
  </style>
</head>
<body>
  <nav>
    <a href="/">Dashboard</a>
    <a href="/paper">Paper Trade (Training)</a>
    <a href="/logs" class="active">Logs</a>
  </nav>
  <h1>Bot Logs (last {{ lines|length }} lines)</h1>
  <pre>{% for line in lines %}<span class="{{ 'line-error' if 'ERROR' in line or 'CRITICAL' in line else ('line-warn' if 'WARNING' in line else '') }}">{{ line }}</span>
{% endfor %}</pre>
</body>
</html>
"""


@app.route("/")
@require_auth
def index():
    return render_template_string(PAGE, state=get_state(), auth_configured=_auth_configured())


@app.route("/paper")
@require_auth
def paper_page():
    return render_template_string(PAPER_PAGE, state=get_state())


@app.route("/logs")
@require_auth
def logs_page():
    n = request.args.get("n", 200, type=int)
    n = max(10, min(n, 2000))
    lines = _tail_log(n)
    return render_template_string(LOGS_PAGE, lines=lines)


def _tail_log(n: int):
    if not os.path.exists(BOT_LOG_PATH):
        return [f"(no log file found at {BOT_LOG_PATH})"]
    try:
        with open(BOT_LOG_PATH, "r", errors="ignore") as f:
            return [l.rstrip("\n") for l in f.readlines()[-n:]]
    except Exception as e:
        return [f"(failed to read log: {e})"]


@app.route("/api/logs")
@require_auth
def api_logs():
    n = request.args.get("n", 200, type=int)
    n = max(10, min(n, 2000))
    return jsonify({"lines": _tail_log(n)})


@app.route("/api/health")
@require_auth
def api_health():
    s = get_state()
    return jsonify({
        "status": "halted" if s.halted else "running",
        "halt_reason": s.halt_reason,
        "uptime_seconds": round(s.uptime_seconds, 1),
        "started_at": s.started_at,
        "mode": s.mode,
        "training_mode": s.training_mode,
        "ml_prediction_enabled": s.ml_prediction_enabled,
        "brti_price": s.brti_price,
        "brti_tick_count": s.brti_tick_count,
        "lifecycle_active_windows": len(s.lifecycle_active_windows),
        "tte_models_fitted": s.tte_models_fitted,
        "tte_total_models": s.tte_total_models,
    })


@app.route("/api/state")
@require_auth
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
        "brti": {
            "price": s.brti_price,
            "spread_bps": s.brti_spread_bps,
            "exchanges_used": s.brti_exchanges_used,
            "tick_count": s.brti_tick_count,
        },
        "arbitrage": {
            "pnl": s.arb_pnl,
            "open_positions": s.arb_open_positions,
            "closed_positions": s.arb_closed_positions,
            "win_rate": s.arb_win_rate,
            "opportunities": s.arb_opportunities,
        },
        "lifecycle": {
            "active_windows": s.lifecycle_active_windows,
            "total_pnl": s.lifecycle_total_pnl,
            "win_rate": s.lifecycle_win_rate,
            "total_trades": s.lifecycle_total_trades,
        },
        "model_accuracy": s.model_accuracy,
        "trade_log_recent": s.trade_log_recent,
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
@require_auth
def api_withdraw():
    """Manual withdrawal endpoint. Triggers USDC transfer from the trading wallet."""
    data = request.get_json()
    amount = data.get("amount", 0)
    destination = data.get("destination", "")

    if amount <= 0:
        return jsonify({"success": False, "message": "Amount must be > $0"})

    # Import here to avoid circular imports
    from alerts.notifier import Severity, notifier

    # We need to signal to the main loop to execute the withdrawal
    # Store the request in a shared state that main.py picks up
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
    if not _auth_configured():
        import logging
        logging.getLogger(__name__).warning(
            "DASHBOARD_USERNAME/DASHBOARD_PASSWORD not set — dashboard has NO "
            "authentication. Fine for localhost-only access; do not expose "
            "this port publicly without setting them."
        )
    app.run(host=settings.dashboard_host, port=settings.dashboard_port)


if __name__ == "__main__":
    run()
