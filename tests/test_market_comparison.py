"""
Unit tests for the implied volatility solver.
Uses synthetic prices generated from Black-Scholes itself, so we know the
"true" answer exactly -- if we price at sigma=0.25 and solve, we should
recover 0.25.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.black_scholes import BlackScholes
from models.market_comparison import implied_volatility


def approx_equal(a, b, tol=1e-3):
    return abs(a - b) < tol


def test_recovers_known_vol_call():
    true_sigma = 0.25
    S, K, T, r = 100, 100, 1, 0.05
    market_price = BlackScholes(S, K, T, r, true_sigma, "call").price()
    iv = implied_volatility(market_price, S, K, T, r, "call")
    assert approx_equal(iv, true_sigma), f"Got {iv}, expected {true_sigma}"


def test_recovers_known_vol_put():
    true_sigma = 0.35
    S, K, T, r = 100, 110, 0.5, 0.03
    market_price = BlackScholes(S, K, T, r, true_sigma, "put").price()
    iv = implied_volatility(market_price, S, K, T, r, "put")
    assert approx_equal(iv, true_sigma), f"Got {iv}, expected {true_sigma}"


def test_recovers_known_vol_otm_call():
    # OTM options are the harder case numerically -- low prices, flatter vega
    true_sigma = 0.40
    S, K, T, r = 100, 130, 0.25, 0.05
    market_price = BlackScholes(S, K, T, r, true_sigma, "call").price()
    iv = implied_volatility(market_price, S, K, T, r, "call")
    assert approx_equal(iv, true_sigma, tol=1e-2), f"Got {iv}, expected {true_sigma}"


def test_returns_nan_for_impossible_price():
    # A price below intrinsic value has no valid implied vol
    iv = implied_volatility(market_price=0.0001, S=100, K=50, T=1, r=0.05, option_type="call")
    import math
    assert math.isnan(iv)


if __name__ == "__main__":
    tests = [
        test_recovers_known_vol_call,
        test_recovers_known_vol_put,
        test_recovers_known_vol_otm_call,
        test_returns_nan_for_impossible_price,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__} -> {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
