#!/usr/bin/env python3
"""Tests for the Blahut--Arimoto capacity computation.

Run with pytest (``pytest test_arimoto.py``) or directly as a script
(``python3 test_arimoto.py``) -- the ``__main__`` block executes every
``test_*`` function and needs nothing beyond NumPy.
"""

from __future__ import annotations

import math

import numpy as np

from simulate_arimoto import (
    CHANNEL,
    arimoto,
    kkt_gaps,
    mutual_information,
    run_to_convergence,
)

# Values committed in the result tables (nats), used as a regression guard.
KNOWN_CURRENT = {0: 0.085819314, 1: 0.087756916, 10: 0.099720339, 114: 0.112019969}
KNOWN_UPDATED = {0: 0.086804047, 1: 0.088668035, 7: 0.097211250}


def test_channel_is_valid_dmc() -> None:
    assert np.allclose(CHANNEL.sum(axis=0), 1.0), "each p(.|x_j) must be a distribution"
    assert np.all(CHANNEL >= 0.0)


def test_mutual_information_matches_h_y_minus_h_y_given_x() -> None:
    """Cross-check I(X;Y) against the H(Y) - H(Y|X) decomposition."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.dirichlet(np.ones(CHANNEL.shape[1]))
        q = CHANNEL @ p
        h_y = -np.sum(q * np.log(q))
        col_entropy = -np.sum(np.where(CHANNEL > 0, CHANNEL * np.log(CHANNEL), 0.0), axis=0)
        h_y_given_x = float(col_entropy @ p)
        assert math.isclose(
            mutual_information(CHANNEL, p), h_y - h_y_given_x, abs_tol=1e-12
        )


def test_capacity_increases_monotonically() -> None:
    history = arimoto(CHANNEL, max_t=200)
    for h in history:
        # C(t,t) <= C(t+1,t): the p-update never decreases capacity.
        assert h.capacity_updated >= h.capacity_current - 1e-12
    for prev, nxt in zip(history, history[1:]):
        # C(t+1,t) <= C(t+1,t+1): the phi-update never decreases capacity.
        assert nxt.capacity_current >= prev.capacity_updated - 1e-12


def test_regression_against_committed_tables() -> None:
    history = arimoto(CHANNEL, max_t=114)
    for t, expected in KNOWN_CURRENT.items():
        assert math.isclose(history[t].capacity_current, expected, abs_tol=1e-9)
    for t, expected in KNOWN_UPDATED.items():
        assert math.isclose(history[t].capacity_updated, expected, abs_tol=1e-9)


def test_converges_to_true_capacity_via_kkt() -> None:
    """The converged input satisfies the channel-capacity KKT conditions."""
    p_star, _ = run_to_convergence(CHANNEL)
    capacity, divergence = kkt_gaps(CHANNEL, p_star)
    support = p_star > 1e-9
    assert np.allclose(divergence[support], capacity, atol=1e-9)  # tight on support
    assert np.all(divergence <= capacity + 1e-9)                  # slack off support
    assert math.isclose(capacity, 0.112034669, abs_tol=1e-6)     # known capacity (nats)


def test_redundant_symbol_is_dropped() -> None:
    p_star, _ = run_to_convergence(CHANNEL)
    assert p_star[1] < 1e-6, "x2 carries no extra information and should be unused"
    assert p_star[0] > 0.45 and p_star[2] > 0.45


def test_bsc_matches_closed_form() -> None:
    """Generality check: a binary symmetric channel has the textbook capacity."""
    f = 0.1
    bsc = np.array([[1 - f, f], [f, 1 - f]])
    p_star, _ = run_to_convergence(bsc)
    capacity, _ = kkt_gaps(bsc, p_star)
    h_b = -f * math.log(f) - (1 - f) * math.log(1 - f)
    assert np.allclose(p_star, [0.5, 0.5], atol=1e-9)       # uniform input is optimal
    assert math.isclose(capacity, math.log(2) - h_b, abs_tol=1e-9)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
