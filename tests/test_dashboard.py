"""
Tests for the dashboard: state wiring (compute_model_accuracy) and that the
Flask routes render/respond, including the auth gate.
"""
import json

import pytest

from config.settings import settings
from dashboard.state import compute_model_accuracy


class FakeTradeLogger:
    def __init__(self, rows):
        self._rows = rows

    def get_recent_trades(self, limit=500):
        return self._rows[:limit]


def test_compute_model_accuracy_empty_without_logger():
    assert compute_model_accuracy(None) == {}


def test_compute_model_accuracy_ignores_open_trades_and_unscored_trades():
    rows = [
        {"exit_price": None, "won": 0, "metadata_json": json.dumps({"model_prob": 0.8})},
        {"exit_price": 1.0, "won": 1, "metadata_json": json.dumps({})},  # no model_prob
        {"exit_price": 1.0, "won": 1, "metadata_json": "not json"},
    ]
    result = compute_model_accuracy(FakeTradeLogger(rows))
    assert result == {"n_predictions": 0}


def test_compute_model_accuracy_scores_closed_trades_with_predictions():
    rows = [
        {"exit_price": 1.0, "won": 1, "metadata_json": json.dumps({"model_prob": 0.9})},
        {"exit_price": 0.0, "won": 0, "metadata_json": json.dumps({"model_prob": 0.1})},
    ]
    result = compute_model_accuracy(FakeTradeLogger(rows))
    assert result["n_predictions"] == 2
    # Both predictions were well-calibrated -> low Brier score
    assert result["brier_score"] < 0.05
    assert result["better_than_baseline"] is True


def test_dashboard_auth_blocks_without_credentials(monkeypatch):
    monkeypatch.setattr(settings, "dashboard_username", "admin")
    monkeypatch.setattr(settings, "dashboard_password", "secret")
    import dashboard.app as dapp
    client = dapp.app.test_client()
    r = client.get("/")
    assert r.status_code == 401


def test_dashboard_auth_allows_with_correct_credentials(monkeypatch):
    monkeypatch.setattr(settings, "dashboard_username", "admin")
    monkeypatch.setattr(settings, "dashboard_password", "secret")
    import dashboard.app as dapp
    client = dapp.app.test_client()
    import base64
    creds = base64.b64encode(b"admin:secret").decode()
    r = client.get("/", headers={"Authorization": f"Basic {creds}"})
    assert r.status_code == 200


def test_dashboard_open_without_credentials_configured(monkeypatch):
    monkeypatch.setattr(settings, "dashboard_username", "")
    monkeypatch.setattr(settings, "dashboard_password", "")
    import dashboard.app as dapp
    client = dapp.app.test_client()
    r = client.get("/")
    assert r.status_code == 200
