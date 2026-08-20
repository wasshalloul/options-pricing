"""
Binomial Tree Option Pricing Model (Cox-Ross-Rubinstein)
-----------------------------------------------------------
Prices European and American options using a recombining binomial tree.

Why it matters vs. Black-Scholes:
- Black-Scholes only prices European options (exercise at expiry only)
- American options can be exercised any time up to expiry, which matters
  a lot for puts (and for calls on dividend-paying stocks)
- The binomial tree handles this naturally: at each node we compare
  "hold" value vs. "exercise now" value and take the max

As the number of steps -> infinity, the CRR tree price converges to the
Black-Scholes price for European options. We show this convergence in
plot_convergence.py.
"""

import numpy as np


class BinomialTree:
    def __init__(self, S, K, T, r, sigma, steps=200, option_type="call", exercise="european"):
        """
        Parameters
        ----------
        S : float   Current underlying price
        K : float   Strike price
        T : float   Time to expiry, in years
        r : float   Risk-free rate (annualized, continuous compounding)
        sigma : float   Volatility (annualized)
        steps : int   Number of time steps in the tree
        option_type : str   "call" or "put"
        exercise : str   "european" or "american"
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.steps = steps
        self.option_type = option_type.lower()
        self.exercise = exercise.lower()

        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        if self.exercise not in ("european", "american"):
            raise ValueError("exercise must be 'european' or 'american'")

        # CRR parameters
        self.dt = T / steps
        self.u = np.exp(sigma * np.sqrt(self.dt))       # up factor
        self.d = 1 / self.u                               # down factor
        self.p = (np.exp(r * self.dt) - self.d) / (self.u - self.d)  # risk-neutral prob
        self.discount = np.exp(-r * self.dt)

    def _payoff(self, spot):
        if self.option_type == "call":
            return np.maximum(spot - self.K, 0)
        else:
            return np.maximum(self.K - spot, 0)

    def price(self):
        n = self.steps

        # Terminal spot prices at maturity: S * u^j * d^(n-j) for j = 0..n
        j = np.arange(n + 1)
        terminal_spots = self.S * (self.u ** j) * (self.d ** (n - j))
        values = self._payoff(terminal_spots)

        # Backward induction
        for step in range(n - 1, -1, -1):
            j = np.arange(step + 1)
            # Expected value discounted one step back
            values = self.discount * (self.p * values[1:] + (1 - self.p) * values[:-1])

            if self.exercise == "american":
                spots = self.S * (self.u ** j) * (self.d ** (step - j))
                exercise_value = self._payoff(spots)
                values = np.maximum(values, exercise_value)

        return values[0]


if __name__ == "__main__":
    # American put — this is where early exercise actually matters
    am_put = BinomialTree(S=100, K=100, T=1, r=0.05, sigma=0.2, steps=500,
                           option_type="put", exercise="american")
    eu_put = BinomialTree(S=100, K=100, T=1, r=0.05, sigma=0.2, steps=500,
                           option_type="put", exercise="european")

    print(f"American put price: {am_put.price():.4f}")
    print(f"European put price: {eu_put.price():.4f}")
    print(f"Early exercise premium: {am_put.price() - eu_put.price():.4f}")

    # Calls (no dividends) — American and European should be ~identical here,
    # since early exercise is never optimal for a call with no dividends
    am_call = BinomialTree(S=100, K=100, T=1, r=0.05, sigma=0.2, steps=500,
                            option_type="call", exercise="american")
    eu_call = BinomialTree(S=100, K=100, T=1, r=0.05, sigma=0.2, steps=500,
                            option_type="call", exercise="european")
    print(f"\nAmerican call price: {am_call.price():.4f}")
    print(f"European call price: {eu_call.price():.4f}")
