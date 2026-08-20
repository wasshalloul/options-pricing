"""
Monte Carlo Option Pricing
----------------------------
Prices European options via simulation of Geometric Brownian Motion (GBM)
paths, plus a path-dependent Asian option (average price call/put), which
neither Black-Scholes nor the binomial tree can price directly.

Variance reduction: antithetic variates
For every random path generated using Z, we also generate its "mirror" path
using -Z. Averaging paired paths cancels out some of the sampling noise,
cutting the standard error roughly in half for the same number of draws --
i.e. better accuracy for the same computational cost.
"""

import numpy as np


class MonteCarloOptionPricer:
    def __init__(self, S, K, T, r, sigma, option_type="call",
                 n_paths=100_000, n_steps=252, antithetic=True, seed=42):
        """
        Parameters
        ----------
        S, K, T, r, sigma : as in Black-Scholes
        option_type : "call" or "put"
        n_paths : int   number of simulated price paths (pairs, if antithetic)
        n_steps : int   number of time steps per path (252 ~ daily steps/year)
        antithetic : bool   use antithetic variates for variance reduction
        seed : int   RNG seed for reproducibility
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.antithetic = antithetic
        self.rng = np.random.default_rng(seed)

        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

    def _simulate_paths(self):
        """Simulate GBM paths. Returns array of shape (n_sim_paths, n_steps+1)."""
        dt = self.T / self.n_steps
        drift = (self.r - 0.5 * self.sigma ** 2) * dt
        vol = self.sigma * np.sqrt(dt)

        half = self.n_paths
        Z = self.rng.standard_normal((half, self.n_steps))

        if self.antithetic:
            Z = np.vstack([Z, -Z])  # mirror paths

        log_returns = drift + vol * Z
        log_paths = np.cumsum(log_returns, axis=1)
        log_paths = np.hstack([np.zeros((log_paths.shape[0], 1)), log_paths])
        paths = self.S * np.exp(log_paths)
        return paths

    def _payoff(self, terminal_or_avg):
        if self.option_type == "call":
            return np.maximum(terminal_or_avg - self.K, 0)
        else:
            return np.maximum(self.K - terminal_or_avg, 0)

    def price_european(self):
        """Standard European option, using only the terminal price."""
        paths = self._simulate_paths()
        terminal = paths[:, -1]
        payoffs = self._payoff(terminal)
        discounted = np.exp(-self.r * self.T) * payoffs
        price = discounted.mean()
        std_err = discounted.std(ddof=1) / np.sqrt(len(discounted))
        return price, std_err

    def price_asian(self):
        """
        Asian option: payoff based on the AVERAGE price over the path,
        not just the terminal price. This smooths out volatility, so
        Asian options are cheaper than equivalent European options --
        useful for hedging commodity/FX exposure without paying for
        single-point-in-time volatility.
        """
        paths = self._simulate_paths()
        avg_price = paths[:, 1:].mean(axis=1)  # exclude S0, average over the path
        payoffs = self._payoff(avg_price)
        discounted = np.exp(-self.r * self.T) * payoffs
        price = discounted.mean()
        std_err = discounted.std(ddof=1) / np.sqrt(len(discounted))
        return price, std_err


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from black_scholes import BlackScholes

    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    bs = BlackScholes(S, K, T, r, sigma, "call").price()
    print(f"Black-Scholes (analytic):      {bs:.4f}")

    mc_plain = MonteCarloOptionPricer(S, K, T, r, sigma, "call",
                                       n_paths=50_000, antithetic=False)
    price, se = mc_plain.price_european()
    print(f"Monte Carlo (no variance red.): {price:.4f}  (std err: {se:.4f})")

    mc_anti = MonteCarloOptionPricer(S, K, T, r, sigma, "call",
                                      n_paths=50_000, antithetic=True)
    price, se = mc_anti.price_european()
    print(f"Monte Carlo (antithetic):       {price:.4f}  (std err: {se:.4f})")

    asian = MonteCarloOptionPricer(S, K, T, r, sigma, "call",
                                    n_paths=50_000, antithetic=True)
    a_price, a_se = asian.price_asian()
    print(f"\nAsian call (avg price):         {a_price:.4f}  (std err: {a_se:.4f})")
    print(f"European call:                  {price:.4f}")
    print(f"(Asian is cheaper: averaging dampens volatility)")
