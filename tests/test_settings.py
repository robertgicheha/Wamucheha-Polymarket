"""Sanity checks on config.settings defaults relevant to the ML prediction rollout."""
from config.settings import settings


def test_ml_prediction_disabled_by_default():
    """
    Must default to False: the fallback heuristic should be what runs until
    someone has bootstrapped models and deliberately opted in.
    """
    assert settings.ml_prediction_enabled is False


def test_paper_mode_is_the_default_trading_mode():
    assert settings.trading_mode == "paper"


def test_training_mode_defaults_on():
    assert settings.training_mode is True
