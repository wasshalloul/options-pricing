"""
Unit tests for the binomial tree model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.binomial_tree import BinomialTree
from models.black_scholes import BlackScholes


def approx_equal(a, b, tol=1e-2):
    return abs(a - b) < tol


def test_european_call_converges_to_black_scholes():
    bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    tree = BinomialTree(S=100, K=100, T=1, r=0.05, sigma=0.2, steps=1000,
                         option_type="call", exercise="european")
    assert approx_equal(bs.price(), tree.price(), tol=0.05), (
        f"BS={bs.price()} vs Tree={tree.price()}"
    )


def test_european_put_converges_to_black_scholes():
    bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
    tree = BinomialTree(S=100, K=100, T=1, r=0.05, sigma=0.2, steps=1000,
                         option_type="put", exercise="european")
    assert approx_equal(bs.price(), tree.price(), tol=0.05), (
        f"BS={bs.price()} vs Tree={tree.price()}"
    )


def test_american_put_worth_more_than_european():
    am = BinomialTree(S=100, K=100, T=1, r=0.05, sigma=0.2, steps=500,
                       option_type="put", exercise="american")
    eu = BinomialTree(S=100, K=100, T=1, r=0.05, sigma=0.2, steps=500,
                       option_type="put", exercise="european")
    assert am.price() >= eu.price() - 1e-6


def test_american_call_equals_european_call_no_dividends():
    # With no dividends, early exercise of a call is never optimal
    am = BinomialTree(S=100, K=100, T=1, r=0.05, sigma=0.2, steps=500,
                       option_type="call", exercise="american")
    eu = BinomialTree(S=100, K=100, T=1, r=0.05, sigma=0.2, steps=500,
                       option_type="call", exercise="european")
    assert approx_equal(am.price(), eu.price(), tol=1e-3)


def test_deep_itm_american_put_near_intrinsic():
    # Deep ITM American put should trade very close to intrinsic value
    tree = BinomialTree(S=50, K=100, T=1, r=0.05, sigma=0.2, steps=500,
                         option_type="put", exercise="american")
    intrinsic = 100 - 50
    assert tree.price() >= intrinsic - 1e-6
    assert approx_equal(tree.price(), intrinsic, tol=1.0)


def test_invalid_exercise_type_raises():
    try:
        BinomialTree(100, 100, 1, 0.05, 0.2, exercise="bermudan")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [
        test_european_call_converges_to_black_scholes,
        test_european_put_converges_to_black_scholes,
        test_american_put_worth_more_than_european,
        test_american_call_equals_european_call_no_dividends,
        test_deep_itm_american_put_near_intrinsic,
        test_invalid_exercise_type_raises,
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
