"""
Visualize Monte Carlo simulation: sample price paths, and the effect of
antithetic variates on standard error as the number of paths grows.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from models.monte_carlo import MonteCarloOptionPricer

S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Chart 1: sample simulated paths ---
mc_viz = MonteCarloOptionPricer(S, K, T, r, sigma, "call", n_paths=15, n_steps=252, seed=7)
paths = mc_viz._simulate_paths()
time_axis = np.linspace(0, T, paths.shape[1])

for path in paths[:30]:  # antithetic doubles n_paths, cap display at 30
    axes[0].plot(time_axis, path, linewidth=0.8, alpha=0.7)
axes[0].axhline(K, color="black", linestyle="--", linewidth=1, label=f"Strike = {K}")
axes[0].set_title("Sample Simulated GBM Price Paths")
axes[0].set_xlabel("Time (years)")
axes[0].set_ylabel("Price")
axes[0].legend()

# --- Chart 2: standard error vs number of paths, with/without antithetic ---
path_counts = [500, 1000, 2500, 5000, 10000, 25000, 50000]
se_plain, se_anti = [], []

for n in path_counts:
    plain = MonteCarloOptionPricer(S, K, T, r, sigma, "call", n_paths=n,
                                    antithetic=False, seed=1)
    anti = MonteCarloOptionPricer(S, K, T, r, sigma, "call", n_paths=n,
                                   antithetic=True, seed=1)
    _, sp = plain.price_european()
    _, sa = anti.price_european()
    se_plain.append(sp)
    se_anti.append(sa)

axes[1].plot(path_counts, se_plain, "o-", color="#dc2626", label="No variance reduction")
axes[1].plot(path_counts, se_anti, "o-", color="#16a34a", label="Antithetic variates")
axes[1].set_xscale("log")
axes[1].set_xlabel("Number of simulated paths (log scale)")
axes[1].set_ylabel("Standard error")
axes[1].set_title("Standard Error vs Number of Paths")
axes[1].legend()

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), "monte_carlo_paths.png")
plt.savefig(output_path, dpi=150)
print(f"Saved plot to {output_path}")
