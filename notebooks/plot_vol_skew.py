"""
Plot the implied volatility skew from a saved option chain CSV
(produced by market_comparison.py).

Usage:
    python plot_vol_skew.py --ticker AAPL
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    args = parser.parse_args()

    csv_path = os.path.join(os.path.dirname(__file__), "..", "data",
                             f"{args.ticker}_option_chain.csv")
    df = pd.read_csv(csv_path)

    calls = df[df["type"] == "call"].sort_values("strike")
    puts = df[df["type"] == "put"].sort_values("strike")

    plt.figure(figsize=(9, 6))
    plt.plot(calls["strike"], calls["implied_vol"] * 100, "o-",
              color="#2563eb", label="Calls", markersize=4)
    plt.plot(puts["strike"], puts["implied_vol"] * 100, "o-",
              color="#dc2626", label="Puts", markersize=4)

    plt.xlabel("Strike price")
    plt.ylabel("Implied volatility (%)")
    plt.title(f"Implied Volatility Skew -- {args.ticker}\n"
              f"(Black-Scholes assumes this line would be flat)")
    plt.legend()
    plt.grid(alpha=0.3)

    output_path = os.path.join(os.path.dirname(__file__), f"{args.ticker}_vol_skew.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
