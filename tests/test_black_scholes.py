"""
Unit tests for the Black-Scholes model.
Reference values are well-known textbook results (Hull, Options Futures and
Other Derivatives) so a reviewer can independently verify correctness.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
from models.black_scholes import BlackScholes


def approx_equal(a, b, tol=1e-2):
    return abs(a - b) < tol


def test_call_price_matches_reference():
    # S=100, K=100, T=1, r=5%, sigma=20% -> call ~ 10.4506
    opt = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    assert approx_equal(opt.price(), 10.4506), f"Got {opt.price()}"


def test_put_price_matches_reference():
    # Same params -> put ~ 5.5735
    opt = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
    assert approx_equal(opt.price(), 5.5735), f"Got {opt.price()}"


def test_put_call_parity():
    # C - P = S - K*e^(-rT)
    S, K, T, r, sigma = 100, 95, 0.5, 0.03, 0.25
    call = BlackScholes(S, K, T, r, sigma, "call").price()
    put = BlackScholes(S, K, T, r, sigma, "put").price()
    lhs = call - put
    rhs = S - K * math.exp(-r * T)
    assert approx_equal(lhs, rhs, tol=1e-6), f"Parity violated: {lhs} vs {rhs}"


def test_deep_itm_call_delta_near_one():
    opt = BlackScholes(S=200, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    assert opt.delta() > 0.95


def test_deep_otm_call_delta_near_zero():
    opt = BlackScholes(S=50, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    assert opt.delta() < 0.05


def test_call_gamma_equals_put_gamma():
    # Gamma is identical for calls and puts at the same strike/expiry
    call = BlackScholes(100, 100, 1, 0.05, 0.2, "call")
    put = BlackScholes(100, 100, 1, 0.05, 0.2, "put")
    assert approx_equal(call.gamma(), put.gamma(), tol=1e-8)


def test_invalid_option_type_raises():
    try:
        BlackScholes(100, 100, 1, 0.05, 0.2, "banana")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [
        test_call_price_matches_reference,
        test_put_price_matches_reference,
        test_put_call_parity,
        test_deep_itm_call_delta_near_one,
        test_deep_otm_call_delta_near_zero,
        test_call_gamma_equals_put_gamma,
        test_invalid_option_type_raises,
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
