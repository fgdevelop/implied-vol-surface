# implied-vol-surface
Fetch option chains with yfinance, compute Black–Scholes implied volatilities, interpolate an IV surface across moneyness × maturity, fit a bivariate spline, and produce 3D and 2D visualizations. Caches fetched option data to CSV to avoid repeated API calls.
