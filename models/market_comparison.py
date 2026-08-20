"""
Real Market Comparison & Implied Volatility Skew
----------------------------------------------------
Pulls a live option chain from Yahoo Finance, compares market prices to our
Black-Scholes model, and backs out implied volatility for each strike to
show the volatility skew/smile.

Why this matters:
Black-Scholes assumes constant volatility across all strikes. Real markets
don't behave this way -- out-of-the-money puts are typically priced with
higher implied volatility than at-the-money options (the "skew"), reflecting
market pricing of downside/crash risk. Plotting this skew is one of the
most common ways to show you understand where the theoretical model departs
from reality.

Usage:
    python market_comparison.py --ticker AAPL
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
from models.black_scholes import BlackScholes


def implied_volatility(market_price, S, K, T, r, option_type="call"):
    """
    Back out the implied volatility that makes Black-Scholes match the
    observed market price, using Brent's method (a robust root-finder).
    """
    def objective(sigma):
        return BlackScholes(S, K, T, r, sigma, option_type).price() - market_price

    try:
        # Search between 0.1% and 500% annualized vol -- wide enough to
        # capture anything realistic, narrow enough to converge fast
        return brentq(objective, 1e-3, 5.0, maxiter=200)
    except ValueError:
        # No sign change in that range -- market price is likely below
        # intrinsic value or some other data quality issue. Skip it.
        return np.nan


def fetch_option_chain(ticker_symbol, r=0.05, min_days=14, max_days=60):
    """
    Fetch an option chain for a ticker and compute implied volatility for
    each strike using both calls and puts.

    We deliberately avoid the nearest expiry: very short-dated options
    (days or 0DTE) are thin, illiquid, and prone to stale/garbage quotes
    that produce nonsensical implied vols. Instead we pick the first
    expiry that falls within [min_days, max_days] out, which is typically
    liquid enough for clean data.

    We also filter out strikes with no real bid/ask (illiquid) and clip
    implausible implied vols, since those are data quality issues rather
    than genuine market signal.
    """
    ticker = yf.Ticker(ticker_symbol)
    expirations = ticker.options
    if not expirations:
        raise ValueError(f"No option expirations found for {ticker_symbol}")

    from datetime import datetime
    today = datetime.now()

    chosen_expiry = None
    for exp in expirations:
        days_out = (datetime.strptime(exp, "%Y-%m-%d") - today).days
        if min_days <= days_out <= max_days:
            chosen_expiry = exp
            break
    if chosen_expiry is None:
        # fall back to the closest expiry to the target window
        chosen_expiry = min(
            expirations,
            key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d") - today).days - 30)
        )

    expiry = chosen_expiry
    chain = ticker.option_chain(expiry)

    spot = ticker.history(period="1d")["Close"].iloc[-1]

    days_to_expiry = (datetime.strptime(expiry, "%Y-%m-%d") - today).days
    T = max(days_to_expiry, 1) / 365

    def clean_and_price(row):
        """Return a usable mid price, or None if the quote looks illiquid/stale."""
        bid, ask = row.get("bid", 0), row.get("ask", 0)
        if bid > 0 and ask > 0 and ask >= bid:
            spread_pct = (ask - bid) / ((ask + bid) / 2)
            if spread_pct < 1.0:  # reject only truly absurd spreads
                return (bid + ask) / 2
        # fall back to lastPrice if it's recent-looking (nonzero, sane)
        last = row.get("lastPrice", 0)
        if last > 0:
            return last
        return None

    calls_seen, calls_kept = 0, 0
    puts_seen, puts_kept = 0, 0

    results = []
    for _, row in chain.calls.iterrows():
        calls_seen += 1
        mid = clean_and_price(row)
        if mid is None or mid <= 0:
            continue
        if not (0.7 * spot <= row["strike"] <= 1.3 * spot):
            continue
        iv = implied_volatility(mid, spot, row["strike"], T, r, "call")
        if not np.isnan(iv) and iv < 2.0:
            calls_kept += 1
            results.append({"strike": row["strike"], "type": "call",
                             "market_price": mid, "implied_vol": iv})

    for _, row in chain.puts.iterrows():
        puts_seen += 1
        mid = clean_and_price(row)
        if mid is None or mid <= 0:
            continue
        if not (0.7 * spot <= row["strike"] <= 1.3 * spot):
            continue
        iv = implied_volatility(mid, spot, row["strike"], T, r, "put")
        if not np.isnan(iv) and iv < 2.0:
            puts_kept += 1
            results.append({"strike": row["strike"], "type": "put",
                             "market_price": mid, "implied_vol": iv})

    print(f"Calls: kept {calls_kept}/{calls_seen}  |  Puts: kept {puts_kept}/{puts_seen}")

    df = pd.DataFrame(results)
    if df.empty:
        raise ValueError(
            "No usable option quotes survived filtering. This can happen outside "
            "market hours (stale/zero quotes) or for less liquid tickers. Try "
            "running during US market hours, or try a highly liquid ticker like "
            "SPY, AAPL, or MSFT."
        )
    return df, spot, T, expiry


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--rate", type=float, default=0.05, help="Risk-free rate")
    args = parser.parse_args()

    print(f"Fetching option chain for {args.ticker}...")
    df, spot, T, expiry = fetch_option_chain(args.ticker, r=args.rate)

    print(f"\nSpot price: {spot:.2f}")
    print(f"Nearest expiry: {expiry}  (T = {T:.4f} years)")
    print(f"\n{df.sort_values('strike').to_string(index=False)}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "data",
                             f"{args.ticker}_option_chain.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
