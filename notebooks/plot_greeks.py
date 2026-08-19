"""
Plot the Greeks as a function of spot price for a fixed strike/expiry.
Produces a PNG saved to notebooks/greeks_vs_spot.png
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from models.black_scholes import BlackScholes

K = 100
T = 1.0
r = 0.05
sigma = 0.2

spots = np.linspace(50, 150, 200)

calls = [BlackScholes(S, K, T, r, sigma, "call") for S in spots]

prices = [c.price() for c in calls]
deltas = [c.delta() for c in calls]
gammas = [c.gamma() for c in calls]
vegas = [c.vega() / 100 for c in calls]
thetas = [c.theta() / 365 for c in calls]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(f"Black-Scholes Call Option Greeks vs Spot Price\n(K={K}, T={T}y, r={r}, sigma={sigma})", fontsize=13)

axes[0, 0].plot(spots, prices, color="#2563eb")
axes[0, 0].axvline(K, color="gray", linestyle="--", linewidth=0.8)
axes[0, 0].set_title("Price")
axes[0, 0].set_xlabel("Spot")

axes[0, 1].plot(spots, deltas, color="#16a34a")
axes[0, 1].axvline(K, color="gray", linestyle="--", linewidth=0.8)
axes[0, 1].set_title("Delta")
axes[0, 1].set_xlabel("Spot")

axes[0, 2].plot(spots, gammas, color="#dc2626")
axes[0, 2].axvline(K, color="gray", linestyle="--", linewidth=0.8)
axes[0, 2].set_title("Gamma")
axes[0, 2].set_xlabel("Spot")

axes[1, 0].plot(spots, vegas, color="#9333ea")
axes[1, 0].axvline(K, color="gray", linestyle="--", linewidth=0.8)
axes[1, 0].set_title("Vega (per 1% vol)")
axes[1, 0].set_xlabel("Spot")

axes[1, 1].plot(spots, thetas, color="#ea580c")
axes[1, 1].axvline(K, color="gray", linestyle="--", linewidth=0.8)
axes[1, 1].set_title("Theta (per day)")
axes[1, 1].set_xlabel("Spot")

axes[1, 2].axis("off")

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), "greeks_vs_spot.png")
plt.savefig(output_path, dpi=150)
print(f"Saved plot to {output_path}")
