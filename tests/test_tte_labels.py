"""
Tests for ml.tte_orchestrator.self_relative_labels — the label-construction
fix that replaced comparing every historical row against one fixed constant
strike price with a self-relative "is price higher `tte` steps from now"
label (see ml/tte_orchestrator.py module docstring for why).
"""
import numpy as np
import pytest

from ml.tte_orchestrator import self_relative_labels


def test_monotonically_increasing_series_is_always_up():
    prices = np.arange(1.0, 101.0)  # strictly increasing: 1, 2, ..., 100
    labels = self_relative_labels(prices, tte=5)
    assert len(labels) == len(prices) - 5
    assert np.all(labels == 1.0)


def test_monotonically_decreasing_series_is_always_down():
    prices = np.arange(100.0, 0.0, -1.0)
    labels = self_relative_labels(prices, tte=5)
    assert len(labels) == len(prices) - 5
    assert np.all(labels == 0.0)


def test_flat_series_is_never_up():
    prices = np.full(50, 100.0)
    labels = self_relative_labels(prices, tte=3)
    assert np.all(labels == 0.0)  # strictly-greater-than, ties are "not up"


def test_tte_larger_than_series_returns_empty():
    prices = np.arange(1.0, 11.0)  # 10 points
    labels = self_relative_labels(prices, tte=10)
    assert len(labels) == 0
    labels = self_relative_labels(prices, tte=50)
    assert len(labels) == 0


def test_label_is_self_relative_not_fixed_strike():
    """
    The bug this replaces: labels compared against one constant strike price
    for the WHOLE series, so a long-running uptrend (e.g. BTC over 90 days)
    would make almost every label identical regardless of tte, destroying
    the learning signal. A self-relative label must instead track local
    direction: a series with reversals should produce a mix of 0s and 1s.
    """
    prices = np.concatenate([
        np.linspace(100, 200, 50),   # up
        np.linspace(200, 100, 50),   # down
        np.linspace(100, 200, 50),   # up
    ])
    labels = self_relative_labels(prices, tte=1)
    assert 0.0 in labels
    assert 1.0 in labels


def test_specific_known_values():
    prices = np.array([10.0, 12.0, 8.0, 8.0, 20.0])
    # tte=1: compare price[i] vs price[i+1]
    # (10->12 up=1), (12->8 down=0), (8->8 flat=0), (8->20 up=1)
    labels = self_relative_labels(prices, tte=1)
    np.testing.assert_array_equal(labels, [1.0, 0.0, 0.0, 1.0])
