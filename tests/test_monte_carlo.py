"""
Unit tests for the Monte Carlo option pricer.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.monte_carlo import MonteCarloOptionPricer
from models.black_scholes import BlackScholes


def approx_equal(a, b, tol):
    return abs(a - b) < tol


def test_mc_european_call_converges_to_black_scholes():
    bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call").price()
    mc = MonteCarloOptionPricer(S=100, K=100, T=1, r=0.05, sigma=0.2,
                                 option_type="call", n_paths=50_000, seed=1)
    price, se = mc.price_european()
    # within 3 standard errors is a standard statistical tolerance check
    assert approx_equal(price, bs, tol=3 * se), f"MC={price} BS={bs} SE={se}"


def test_mc_european_put_converges_to_black_scholes():
    bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put").price()
    mc = MonteCarloOptionPricer(S=100, K=100, T=1, r=0.05, sigma=0.2,
                                 option_type="put", n_paths=50_000, seed=1)
    price, se = mc.price_european()
    assert approx_equal(price, bs, tol=3 * se), f"MC={price} BS={bs} SE={se}"


def test_antithetic_reduces_standard_error():
    plain = MonteCarloOptionPricer(S=100, K=100, T=1, r=0.05, sigma=0.2,
                                    option_type="call", n_paths=20_000,
                                    antithetic=False, seed=1)
    anti = MonteCarloOptionPricer(S=100, K=100, T=1, r=0.05, sigma=0.2,
                                   option_type="call", n_paths=20_000,
                                   antithetic=True, seed=1)
    _, se_plain = plain.price_european()
    _, se_anti = anti.price_european()
    assert se_anti < se_plain, f"Antithetic SE ({se_anti}) should be < plain SE ({se_plain})"


def test_asian_call_cheaper_than_european_call():
    mc = MonteCarloOptionPricer(S=100, K=100, T=1, r=0.05, sigma=0.2,
                                 option_type="call", n_paths=30_000, seed=1)
    euro_price, _ = mc.price_european()
    asian_price, _ = mc.price_asian()
    assert asian_price < euro_price, f"Asian ({asian_price}) should be < European ({euro_price})"


def test_zero_volatility_call_equals_discounted_intrinsic():
    # With sigma=0, the path is deterministic: S grows at rate r.
    # Payoff = max(S*e^(rT) - K, 0), discounted back at r.
    import numpy as np
    S, K, T, r = 100, 90, 1, 0.05
    mc = MonteCarloOptionPricer(S=S, K=K, T=T, r=r, sigma=1e-6,
                                 option_type="call", n_paths=1000, seed=1)
    price, _ = mc.price_european()
    expected = np.exp(-r * T) * max(S * np.exp(r * T) - K, 0)
    assert approx_equal(price, expected, tol=0.05), f"Got {price}, expected {expected}"


def test_invalid_option_type_raises():
    try:
        MonteCarloOptionPricer(100, 100, 1, 0.05, 0.2, option_type="banana")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [
        test_mc_european_call_converges_to_black_scholes,
        test_mc_european_put_converges_to_black_scholes,
        test_antithetic_reduces_standard_error,
        test_asian_call_cheaper_than_european_call,
        test_zero_volatility_call_equals_discounted_intrinsic,
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
