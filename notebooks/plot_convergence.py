"""
Plot binomial tree price convergence to the Black-Scholes closed-form price
as the number of steps increases. Also plots the early exercise premium
for an American put across spot prices.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from models.binomial_tree import BinomialTree
from models.black_scholes import BlackScholes

S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

# --- Chart 1: Convergence ---
bs_price = BlackScholes(S, K, T, r, sigma, "call").price()

step_counts = [5, 10, 20, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000]
tree_prices = [
    BinomialTree(S, K, T, r, sigma, steps=n, option_type="call", exercise="european").price()
    for n in step_counts
]

# --- Chart 2: Early exercise premium across spot prices ---
spots = np.linspace(60, 140, 60)
premiums = []
for s in spots:
    am = BinomialTree(s, K, T, r, sigma, steps=300, option_type="put", exercise="american")
    eu = BinomialTree(s, K, T, r, sigma, steps=300, option_type="put", exercise="european")
    premiums.append(am.price() - eu.price())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(step_counts, tree_prices, "o-", color="#2563eb", label="Binomial tree price")
axes[0].axhline(bs_price, color="#dc2626", linestyle="--", label=f"Black-Scholes = {bs_price:.4f}")
axes[0].set_xlabel("Number of steps")
axes[0].set_ylabel("Option price")
axes[0].set_title("Binomial Tree Convergence to Black-Scholes\n(European call, K=100, T=1y)")
axes[0].legend()

axes[1].plot(spots, premiums, color="#16a34a")
axes[1].axvline(K, color="gray", linestyle="--", linewidth=0.8, label="Strike")
axes[1].set_xlabel("Spot price")
axes[1].set_ylabel("Early exercise premium (American - European)")
axes[1].set_title("American Put Early Exercise Premium vs Spot")
axes[1].legend()

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), "binomial_convergence.png")
plt.savefig(output_path, dpi=150)
print(f"Saved plot to {output_path}")
