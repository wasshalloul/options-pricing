# Options Pricing Models

Implementation of European option pricing under the Black-Scholes framework,
with the Greeks and visualizations. Part of a broader quant portfolio
(alongside a backtested trading strategy and a risk modeling / VaR project).

## What's in here

- **Black-Scholes closed-form pricer** for European calls and puts
- **Full set of Greeks**: delta, gamma, vega, theta, rho
- **Unit tests** validating against known reference values and put-call parity
- **Visualization** of how each Greek behaves across spot price
- **Binomial tree (CRR)** pricer supporting both European and American exercise
- **Convergence analysis** showing the tree converging to Black-Scholes as steps increase, plus the early exercise premium for American puts

Planned next: Monte Carlo pricing (including a path-dependent Asian option),
plus a comparison against real market option prices to look at implied
volatility skew.

## Example

```python
from models.black_scholes import BlackScholes

call = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
print(call.summary())
# {'price': 10.4506, 'delta': 0.6368, 'gamma': 0.0188, 'vega': 0.3752, 'theta': -0.0176, 'rho': 0.5323}
```

## Greeks vs Spot Price

![Greeks vs Spot](notebooks/greeks_vs_spot.png)

Gamma and vega peak near the strike (as expected — that's where the option's
value is most sensitive to small moves). Delta transitions smoothly from 0
(deep OTM) to 1 (deep ITM). Theta decay is steepest near-the-money, since
that's where time value is greatest.

## Binomial Tree: American vs European

```python
from models.binomial_tree import BinomialTree

am_put = BinomialTree(S=100, K=100, T=1, r=0.05, sigma=0.2, steps=500,
                       option_type="put", exercise="american")
eu_put = BinomialTree(S=100, K=100, T=1, r=0.05, sigma=0.2, steps=500,
                       option_type="put", exercise="european")

print(am_put.price(), eu_put.price())
# 6.0888 5.5695 -- the ~$0.52 gap is the early exercise premium
```

For calls with no dividends, American and European prices are effectively
identical, since early exercise is never optimal in that case — the model
reproduces this correctly.

![Binomial Convergence](notebooks/binomial_convergence.png)

The left chart shows the tree price converging to the Black-Scholes value as
the number of steps increases (the oscillation at low step counts is a known
characteristic of the CRR tree, not a bug). The right chart shows the early
exercise premium for an American put across spot prices — largest deep
in-the-money, decaying to zero out-of-the-money.

## Running it

```bash
pip install -r requirements.txt

# Run the pricer directly
python models/black_scholes.py

# Run tests
python tests/test_black_scholes.py

# Regenerate the plots
python notebooks/plot_greeks.py
python notebooks/plot_convergence.py
```

## Validation

The model is checked against known textbook values (Hull, *Options, Futures
and Other Derivatives*) and against put-call parity:

```
C - P = S - K * e^(-rT)
```

All 7 Black-Scholes tests pass (`tests/test_black_scholes.py`), and all 6
binomial tree tests pass (`tests/test_binomial_tree.py`), including a check
that the tree converges to Black-Scholes for European options and that
American puts are worth at least as much as their European counterpart.

## Project structure

```
options-pricing/
├── models/
│   ├── black_scholes.py       # Closed-form pricing model + Greeks
│   └── binomial_tree.py       # CRR tree, European + American exercise
├── tests/
│   ├── test_black_scholes.py  # Unit tests
│   └── test_binomial_tree.py  # Unit tests
├── notebooks/
│   ├── plot_greeks.py         # Greeks visualization script
│   ├── greeks_vs_spot.png     # Generated chart
│   ├── plot_convergence.py    # Convergence visualization script
│   └── binomial_convergence.png  # Generated chart
├── data/                      # (reserved for market data comparison, next step)
├── requirements.txt
└── README.md
```
