"""
Black-Scholes Option Pricing Model
-----------------------------------
Closed-form pricing for European call and put options, plus the Greeks.

Assumptions:
- No dividends (can be extended with a continuous dividend yield q)
- Constant volatility and risk-free rate
- Log-normal distribution of returns (Geometric Brownian Motion)
"""

import numpy as np
from scipy.stats import norm


class BlackScholes:
    def __init__(self, S, K, T, r, sigma, option_type="call"):
        """
        Parameters
        ----------
        S : float   Current underlying price
        K : float   Strike price
        T : float   Time to expiry, in years
        r : float   Risk-free rate (annualized, continuous compounding)
        sigma : float   Volatility (annualized)
        option_type : str   "call" or "put"
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()

        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

    def _d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )

    def _d2(self):
        return self._d1() - self.sigma * np.sqrt(self.T)

    def price(self):
        d1 = self._d1()
        d2 = self._d2()

        if self.option_type == "call":
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

    # ---- Greeks ----

    def delta(self):
        d1 = self._d1()
        return norm.cdf(d1) if self.option_type == "call" else norm.cdf(d1) - 1

    def gamma(self):
        d1 = self._d1()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        # Per 1.00 change in vol (i.e. 100 vol points). Divide by 100 for per-1% change.
        d1 = self._d1()
        return self.S * norm.pdf(d1) * np.sqrt(self.T)

    def theta(self):
        # Per year. Divide by 365 for per-day decay.
        d1 = self._d1()
        d2 = self._d2()
        term1 = -(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))

        if self.option_type == "call":
            term2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            return term1 + term2
        else:
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
            return term1 + term2

    def rho(self):
        # Per 1.00 change in r. Divide by 100 for per-1% change.
        d2 = self._d2()
        if self.option_type == "call":
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)

    def summary(self):
        return {
            "price": round(self.price(), 4),
            "delta": round(self.delta(), 4),
            "gamma": round(self.gamma(), 4),
            "vega": round(self.vega() / 100, 4),   # per 1% vol
            "theta": round(self.theta() / 365, 4), # per day
            "rho": round(self.rho() / 100, 4),     # per 1% rate
        }


if __name__ == "__main__":
    # Quick sanity check example
    opt = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    print("Call summary:", opt.summary())

    put = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
    print("Put summary:", put.summary())
