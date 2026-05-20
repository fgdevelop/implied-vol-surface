IV Surface — Interpolate and plot implied volatility surfaces for an equity

Overview

This script (`iv_surface.py`) fetches option chain data for an equity using yfinance, computes Black–Scholes implied volatilities for options, interpolates those IVs onto a regular moneyness × maturity mesh, fits a bivariate spline, and produces visualizations:

- a 3D IV surface saved to `volatility_surface.png`
- 2D cross-sections (smiles and term structure) saved to `cross_sections.png`

It also caches fetched option data to CSV to avoid repeated API calls.

Requirements

- Python 3.9+ recommended
- The script depends on:
  - yfinance
  - pandas
  - numpy
  - matplotlib
  - scipy
  - colorama

A `requirements.txt` with minimal conservative pins is provided in the same folder; install with pip.

Install (PowerShell)

```powershell
# From the folder containing requirements.txt
pip install -r "requirements.txt"
```

Quick usage — run as a script (PowerShell)

```powershell
python "iv_surface.py"
```

This will:
- fetch option chains via yfinance (cached CSV named like `opt_data_<TICKER>_..._<DATE>.csv`)
- compute implied volatilities using a Brent root-finder
- interpolate and smooth the IV surface
- save `volatility_surface.png` and `cross_sections.png`
- display the plots interactively

Files produced

- `volatility_surface.png` — 3D IV surface image
- `cross_sections.png` — two-panel figure: volatility smiles and term-structure
- `opt_data_<TICKER>_<...>_<DATE>.csv` — cached option data (generated automatically in the working directory)

Key behavior & notes

- Caching: fetched option data is saved to a CSV file named with the ticker, option type, strike bounds and fetch date. If a cache file exists for the current date, the script loads it instead of redownloading.
- IV calculation: The script computes IV by numerically solving for sigma using Brent’s method. Options with T <= 0 or market price <= 0 are skipped (NaN).
- Interpolation: `griddata` is used to interpolate on a regular moneyness (strike/spot) × time-to-maturity grid. Small NaN holes are selectively filled with nearest-neighbor values if they are near observed data; the final mesh receives mild Gaussian smoothing.
- 3D axes: Matplotlib’s 3D plotting is used; in some linters type hints for 3D axes can show static warnings (these are benign).
- Numerical sensitivity: root-finding limits and interpolation choices influence results. For some strike/maturity combinations, implied vol may not be found or returned as NaN. The script prints messages for successful and failed IV solves.

Configuration / Programmatic usage

The script contains a few classes and helpers you can call from other code:

- `MeshConfig`
  - `min_strike`, `max_strike`: strike bounds (float)
  - `min_iv`, `max_iv`: IV bounds used to filter data (float)
  - `nM`, `nT`: integers for the number of grid points on moneyness and maturity axes
  - `interp_method`: one of `'linear'`, `'cubic'`, `'nearest'`

- `OptMktData.get_option_data(eq_ticker, risk_free_rate, mesh_config, opt_type, liquidity_flag=False)`
  - Fetches option chain data from yfinance, computes expiry (time-to-maturity), mid-market price, spot and returns a DataFrame with columns such as `strike`, `opt_mkt_price`, `option_type`, `expiry`, `spot`, `risk_free_rate`, `moneyness`
  - `liquidity_flag=True` includes OTM calls and puts to generate a "liquid" cross-section

- `CalculateIV.calculate_iv(opt_price_mkt, S, K, r, T, opt_type)`
  - Returns implied sigma (annualized volatility) or NaN on failure

- `PlotIV.create_mesh_and_interpolate(df_opt_data, mesh_config)`
  - Returns `MeshOutput` containing the mesh arrays and `spline_fit` (`SmoothBivariateSpline`)

- `PlotIV.plot_iv_surface(...)` and `PlotIV.plot_iv_smile(...)` and `PlotIV.plot_term_structure(...)`
  - Produce the figure panels. You can call these with your own Matplotlib axes to integrate into larger figures.

Example programmatic snippet

```python
from projects_for_linkedin.iv_surface import MeshConfig, OptMktData, CalculateIV, PlotIV

mesh_cfg = MeshConfig(min_strike=50, max_strike=400, min_iv=0.0, max_iv=1.0, nM=60, nT=40, interp_method='linear')
df = OptMktData.get_option_data('AAPL', 0.01, mesh_cfg, 'CALL', liquidity_flag=False)
df['implied_volatility'] = df.apply(lambda r: CalculateIV.calculate_iv(r['opt_mkt_price'], r['spot'], r['strike'], r['risk_free_rate'], r['expiry'], r['option_type']), axis=1)
mesh_out = PlotIV.create_mesh_and_interpolate(df[~df['implied_volatility'].isnull()], mesh_cfg)
PlotIV.plot_iv_surface('AAPL', 'CALL', mesh_cfg, mesh_out)
```

Recommended `.gitignore` additions

Add to `.gitignore` to avoid committing generated images and cached CSVs:

```
# IV surface outputs and caches
volatility_surface.png
cross_sections.png
opt_data_*.csv
```

Troubleshooting

- If plotting raises backend issues on headless systems, configure matplotlib backend (e.g., Agg) or run in an environment with a display.
- If yfinance calls fail, check network connectivity and rate-limiting; cached CSV reduces repeated API calls.
- If many IV solves return NaN, check that `opt_mkt_price` (mid) is reasonable and T > 0.

License & contribution

- This README does not add a license. If you plan to publish on GitHub, add a LICENSE file (e.g., MIT) as appropriate.
- Contributions: open a PR with improvements, include unit tests for numerical pieces (IV solver, interpolation).

