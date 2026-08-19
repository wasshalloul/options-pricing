# Options Pricing Models

Implementation of European option pricing under the Black-Scholes framework,
with the Greeks and visualizations. Part of a broader quant portfolio
(alongside a backtested trading strategy and a risk modeling / VaR project).

## What's in here

- **Black-Scholes closed-form pricer** for European calls and puts
- **Full set of Greeks**: delta, gamma, vega, theta, rho
- **Unit tests** validating against known reference values and put-call parity
- **Visualization** of how each Greek behaves across spot price

Planned next: binomial tree (American options) and Monte Carlo pricing
(including a path-dependent Asian option), plus a comparison against real
market option prices to look at implied volatility skew.

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

## Running it

```bash
pip install -r requirements.txt

# Run the pricer directly
python models/black_scholes.py

# Run tests
python tests/test_black_scholes.py

# Regenerate the plot
python notebooks/plot_greeks.py
```

## Validation

The model is checked against known textbook values (Hull, *Options, Futures
and Other Derivatives*) and against put-call parity:

```
C - P = S - K * e^(-rT)
```

All 7 unit tests pass — see `tests/test_black_scholes.py`.

## Project structure

```
options-pricing/
├── models/
│   └── black_scholes.py       # Core pricing model + Greeks
├── tests/
│   └── test_black_scholes.py  # Unit tests
├── notebooks/
│   ├── plot_greeks.py         # Visualization script
│   └── greeks_vs_spot.png     # Generated chart
├── data/                      # (reserved for market data comparison, next step)
├── requirements.txt
└── README.md
```
