from os.path import exists
import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from dataclasses import dataclass
from colorama import Fore

from scipy.stats import norm
from scipy.optimize import brentq
from scipy.spatial import cKDTree
from scipy.interpolate import griddata, SmoothBivariateSpline
from scipy.ndimage import gaussian_filter

@dataclass
class MeshConfig:
    """Configuration for the interpolation mesh used to build the IV surface.

    Attributes:
        min_strike (float): Minimum strike to include in the mesh.
        max_strike (float): Maximum strike to include in the mesh.
        min_iv (float): Minimum implied volatility to accept.
        max_iv (float): Maximum implied volatility to accept.
        nM (int): Number of points on the moneyness axis.
        nT (int): Number of points on the maturity (time) axis.
        interp_method (str): Interpolation method for griddata ('linear', 'cubic', 'nearest').
    """

    def __init__(self,
                 min_strike: float,
                 max_strike: float,
                 min_iv: float,
                 max_iv: float,
                 nM: int,
                 nT: int,
                 interp_method: str
                 ):
        self.min_strike = min_strike
        self.max_strike = max_strike
        self.min_iv = min_iv
        self.max_iv = max_iv
        self.nM = nM
        self.nT = nT
        self.interp_method = interp_method

@dataclass
class MeshOutput:
    """Container for the mesh and fit results produced by interpolation.

    Attributes:
        maturity_mesh (np.ndarray): 2-D array of maturities for each mesh point.
        moneyness_mesh (np.ndarray): 2-D array of moneyness values for each mesh point.
        iv_mesh (np.ndarray): 2-D array of interpolated implied volatilities.
        maturity_grid (np.ndarray): 1-D maturity grid used to construct the mesh.
        moneyness_grid (np.ndarray): 1-D moneyness grid used to construct the mesh.
        spline_fit (SmoothBivariateSpline): Bivariate spline fit to the raw IV data.
    """

    def __init__(self,
                 maturity_mesh: np.ndarray,
                 moneyness_mesh: np.ndarray,
                 iv_mesh: np.ndarray,
                 maturity_grid: np.ndarray,
                 moneyness_grid: np.ndarray,
                 spline_fit: SmoothBivariateSpline
                 ):
        self.maturity_mesh = maturity_mesh
        self.moneyness_mesh = moneyness_mesh
        self.iv_mesh = iv_mesh
        self.maturity_grid = maturity_grid
        self.moneyness_grid = moneyness_grid
        self.spline_fit = spline_fit


class OptMktData:
    """Helpers to fetch and prepare option market data from yfinance.

    Methods in this class download option chains for a ticker, compute
    time-to-maturity, filter by strike and liquidity, compute mid-market
    prices and assemble a dataframe ready for implied volatility estimation.
    """

    @staticmethod
    def calculate_maturity_date(ref_date: dt.datetime, maturity_date: str) -> float:
        """Convert a string expiry date to time-to-maturity in years.

        The function parses `maturity_date` (format YYYY-MM-DD) and computes the
        time difference from `ref_date` expressed in trading-year units (252 days).

        Args:
            ref_date (datetime.datetime): Reference date (typically today).
            maturity_date (str): Expiry date string in the format '%Y-%m-%d'.

        Returns:
            float: Time to maturity in years (days/252). May be negative if expiry is before ref_date.
        """

        expiry = dt.datetime.strptime(maturity_date, "%Y-%m-%d")

        return (expiry - ref_date).days / 252

    @staticmethod
    def get_option_data(eq_ticker: str,
                        risk_free_rate: float,
                        mesh_config: MeshConfig,
                        opt_type: str,
                        liquidity_flag: bool = False) -> pd.DataFrame:
        """Fetch option chain data for an equity and prepare a dataframe for IV calculations.

        The function uses yfinance to fetch option chains for all expiries available
        on the ticker, computes time-to-maturity, filters strikes according to
        `mesh_config`, optionally filters only liquid OTM options, computes a mid
        market price and stores spot and risk-free rate information. Results are
        cached to a CSV file named according to the ticker, opt type and mesh range.

        Args:
            eq_ticker (str): Equity ticker symbol (e.g. 'AAPL').
            risk_free_rate (float): Annual risk-free rate (decimal, e.g. 0.01 for 1%).
            mesh_config (MeshConfig): Mesh configuration with strike and IV bounds.
            opt_type (str): Option type to include ('CALL' or 'PUT').
            liquidity_flag (bool): If True, include both CALL and PUT OTM options
                to gather a broader (liquid) cross-section.

        Returns:
            pandas.DataFrame: Prepared option market data with columns including
                'strike', 'opt_mkt_price', 'option_type', 'expiry', 'spot',
                'risk_free_rate', and 'moneyness'. The dataframe is also cached
                to disk and loaded from cache on subsequent runs.
        """

        # Get equity info, list of available expirations
        ticker = yf.Ticker(eq_ticker)
        list_options_dates = ticker.options

        # Get current ref date for maturity calculation
        ref_date = dt.datetime.now()

        file_name = (
            f"opt_data_{eq_ticker}_{opt_type.upper()}_"
            f"liq{liquidity_flag}_"
            f"k_{mesh_config.min_strike:.0f}to{mesh_config.max_strike:.0f}_"
            f"{ref_date:%Y-%m-%d}.csv"
        )
        if exists(file_name):
            print(f"Cache Hit! Loading option data from local file: {file_name}")
            return pd.read_csv(file_name)
        print(f"Cache Miss! Fetching live data from yfinance API for {eq_ticker}...")

        # Gather the underlying asset spot
        spot = ticker.history(period="1d")['Close'].iloc[-1]
        print(f"{eq_ticker} spot at {ref_date:%Y-%m-%d} is given by {spot:,.2f}.")

        df_option_data = pd.DataFrame()
        for maturity_date in list_options_dates:

            # Get options for specific maturity date
            opt_chain_for_maturity = ticker.option_chain(maturity_date)

            # Calculate maturity, if the maturity date is negative or to short, ignore it
            T = OptMktData.calculate_maturity_date(ref_date, maturity_date)
            if T <= 0.01:
                continue

            # Extract relevant call option data, labeling it
            df_call_data = opt_chain_for_maturity.calls[['strike', 'lastPrice', 'bid', 'ask', 'volume']].copy()
            df_call_data['mid'] = (df_call_data['bid'] + df_call_data['ask']) / 2
            df_call_data['option_type'] = 'CALL'
            df_call_data['expiry'] = T

            # Extract relevant put option data, labeling it
            df_put_data = opt_chain_for_maturity.puts[['strike', 'lastPrice', 'bid', 'ask', 'volume']].copy()
            df_put_data['mid'] = (df_put_data['bid'] + df_put_data['ask']) / 2
            df_put_data['option_type'] = 'PUT'
            df_put_data['expiry'] = T

            # Concatenate call and put dataframes for specific maturity
            df_option_data = pd.concat([df_option_data, df_call_data, df_put_data], ignore_index=True)

        # Filter only liquid options (PUT and CALL OTM)
        if liquidity_flag:
            call_otm_mask, put_otm_mask = (df_option_data.strike > spot), (df_option_data.strike < spot)
            df_liquid_call = df_option_data[(df_option_data.option_type == 'CALL') & call_otm_mask]
            df_liquid_put = df_option_data[(df_option_data.option_type == 'PUT') & put_otm_mask]
            df_option_data = pd.concat([df_liquid_call, df_liquid_put], ignore_index=True)

        # Filter options dataframe keeping only the ones with strike within the specified range
        df_option_data = df_option_data[
            (mesh_config.min_strike <= df_option_data.strike) &
            (df_option_data.strike <= mesh_config.max_strike)
        ].reset_index(drop=True)
        if opt_type.upper() not in ['CALL', 'PUT']:
            raise ValueError(f"Option type must be 'CALL' or 'PUT', {opt_type} is not available.")

        # Filter options dataframe keeping only one option type
        if not liquidity_flag:
            df_option_data = df_option_data[df_option_data.option_type.isin([opt_type.upper()])]

        # Include spot price and risk-free rate into options dataframe
        df_option_data['spot'] = spot
        df_option_data['risk_free_rate'] = risk_free_rate
        df_option_data['moneyness'] = df_option_data['strike'] / spot
        df_option_data.rename(columns={'mid': 'opt_mkt_price'}, inplace=True)

        df_option_data.to_csv(file_name, index=False)
        print(f"Successfully cached option data to local file: {file_name}.")

        return df_option_data

class CalculateIV:
    """Black-Scholes pricing helpers and implied volatility solver.

    This class groups static methods to compute Black-Scholes call/put prices
    and to numerically invert market option prices to implied volatilities
    using a root-finding method (Brent's method).
    """

    @staticmethod
    def call_bs_price(S: float, K: float, r: float, T: float, sigma: float) -> float:
        """Compute the Black-Scholes European call price.

        Args:
            S (float): Underlying spot price.
            K (float): Option strike price.
            r (float): Continuously compounded risk-free rate.
            T (float): Time to maturity in years.
            sigma (float): Volatility (annualized) used in pricing.

        Returns:
            float: Black-Scholes call option price.
        """

        d_1 = (np.log(S/K) + (0.5*sigma**2 + r)*T)/(sigma*np.sqrt(T))
        d_2 = d_1 - sigma*np.sqrt(T)
        call_price = S * norm.cdf(d_1) - K * np.exp(-r*T) * norm.cdf(d_2)

        return call_price

    @staticmethod
    def put_bs_price(S: float, K: float, r: float, T: float, sigma: float) -> float:
        """Compute the Black-Scholes European put price.

        Args:
            S (float): Underlying spot price.
            K (float): Option strike price.
            r (float): Continuously compounded risk-free rate.
            T (float): Time to maturity in years.
            sigma (float): Volatility (annualized) used in pricing.

        Returns:
            float: Black-Scholes put option price.
        """

        d_1 = (np.log(S / K) + (0.5 * sigma ** 2 + r) * T) / (sigma * np.sqrt(T))
        d_2 = d_1 - sigma * np.sqrt(T)
        put_price = K * np.exp(-r * T) * (1 - norm.cdf(d_2)) - S * (1 - norm.cdf(d_1))

        return put_price

    @staticmethod
    def calculate_iv(opt_price_mkt: float, S: float, K: float, r: float, T: float, opt_type: str) -> float:
        """
        Calculate implied volatility through Brent's method,
        returning NaN for invalid values (negative maturity or market prices and non-convergent values).
        """

        # Check for valid maturity and option market price
        if T <= 0 or opt_price_mkt <= 0:
            return np.nan

        # Define root finding function to apply the numerical method
        dict_bs_minus_market_price = {
            'CALL': CalculateIV.call_bs_price,
            'PUT': CalculateIV.put_bs_price
        }
        target = lambda sigma: dict_bs_minus_market_price[opt_type](S, K, r, T, sigma) - opt_price_mkt

        try:
            # Apply Brent method to find the implied volatility for specific option
            implied_sigma = brentq(target, 1e-3, 5.0, maxiter=100)
            print(f'IV successfully calculated for option (K, T, V) = ({K:,.2f}, {T:,.4f}, {opt_price_mkt:,.2f}), option type {opt_type}, yielding IV = {implied_sigma*100:,.2f} %.')

        except ValueError as e:
            # If a ValueError occurs, use NaN instead
            print(Fore.RED + f'IV not calculated for option (K, T, V) = ({K:,.2f}, {T:,.4f}, {opt_price_mkt:,.2f}), option type {opt_type}, Error: {e}.' + Fore.RESET)
            return np.nan

        return implied_sigma

class PlotIV:
    """Utilities to interpolate, smooth and plot implied volatility surfaces.

    This class provides methods to build a regular mesh from scattered IV
    observations, fill small holes, fit a SmoothBivariateSpline and create
    both 3D surface visualizations and 2D cross-sections (smiles and term
    structures).
    """

    @staticmethod
    def fill_mesh_nan_with_nearest_interp(M: np.ndarray,
                                          T: np.ndarray,
                                          IV: np.ndarray,
                                          K_mesh: np.ndarray,
                                          T_mesh: np.ndarray,
                                          IV_mesh: np.ndarray,
                                          nan_mask: np.ndarray) -> np.ndarray:
        """Fill NaN positions in the interpolated IV mesh using nearest-neighbour.

        Only fills NaN cells whose nearest data point lies within a conservative
        threshold (to avoid extrapolating far beyond the convex hull of data).

        Args:
            M (np.ndarray): 1-D array of moneyness samples from raw data.
            T (np.ndarray): 1-D array of maturities corresponding to raw data.
            IV (np.ndarray): 1-D array of implied volatilities corresponding to (M, T).
            K_mesh (np.ndarray): 2-D moneyness mesh.
            T_mesh (np.ndarray): 2-D maturity mesh.
            IV_mesh (np.ndarray): 2-D interpolated IV mesh (may contain NaNs).
            nan_mask (np.ndarray): Boolean mask where IV_mesh is NaN.

        Returns:
            np.ndarray: IV mesh with selected NaNs filled by nearest interpolation.
        """

        # Fill only where nearest is within convex hull (small holes)
        # Use a conservative mask: fill only positions whose nearest interpolation is not too far
        IV_near = griddata((M, T), IV, (K_mesh, T_mesh), method='nearest')

        # compute distance to nearest data point quickly:
        pts = np.vstack([M, T]).T
        tree = cKDTree(pts)
        query_pts = np.vstack([K_mesh[nan_mask].ravel(), T_mesh[nan_mask].ravel()]).T
        dists, _ = tree.query(query_pts, k=1)

        # threshold (in strike/maturity units) - adjust if needed
        fill_threshold = (M.max() - M.min()) * 0.05 + (T.max() - T.min()) * 0.05
        fill_allowed = dists < fill_threshold

        # apply fill selectively
        IV_mesh_flat = IV_mesh.ravel()
        nan_idx_flat = np.flatnonzero(nan_mask.ravel())
        to_fill_idx = nan_idx_flat[fill_allowed]
        IV_mesh_flat[to_fill_idx] = IV_near.ravel()[to_fill_idx]
        IV_mesh = IV_mesh_flat.reshape(IV_mesh.shape)

        return IV_mesh

    @staticmethod
    def create_mesh_and_interpolate(df_opt_data: pd.DataFrame, mesh_config: MeshConfig) -> MeshOutput:
        """Create a regular moneyness/maturity mesh and interpolate implied vols.

        This function takes a dataframe with columns ['moneyness', 'expiry',
        'implied_volatility'], filters out-of-bound IV values, interpolates the
        IV surface onto a regular grid using scipy.griddata and fits a
        SmoothBivariateSpline to the raw (M,T,IV) points for later queries.

        Small holes in the interpolated grid are filled using a nearest-neighbour
        approach and the final mesh is slightly smoothed with a gaussian filter.

        Args:
            df_opt_data (pd.DataFrame): Input option data with implied vol column.
            mesh_config (MeshConfig): Mesh configuration specifying grids and bounds.

        Returns:
            MeshOutput: Container with mesh arrays and the fitted spline object.
        """

        # Rename and ensure columns
        df_plot = df_opt_data.copy()
        df_plot = df_plot[['moneyness', 'expiry', 'implied_volatility']].dropna()

        # Keep only "reasonable" rows (optional), wide bounds, just to remove junk
        df_plot = df_plot[
            (df_plot['implied_volatility'] >= mesh_config.min_iv) &
            (df_plot['implied_volatility'] <= mesh_config.max_iv)
            ].reset_index(drop=True)

        # Interpolate on a grid (linear, conservative)
        M = df_plot['moneyness'].values
        T = df_plot['expiry'].values
        IV = df_plot['implied_volatility'].values

        # Initialize mesh of xy-plane points
        M_grid = np.linspace(M.min(), M.max(), mesh_config.nM)
        T_grid = np.linspace(T.min(), T.max(), mesh_config.nT)
        M_mesh, T_mesh = np.meshgrid(M_grid, T_grid)

        # Interpolate implied volatility for each moneyness and maturity value using specified method
        possible_interp = ('cubic', 'linear', 'nearest')
        if mesh_config.interp_method not in possible_interp:
            raise ValueError(f'The interpolation method must be one of the following: {possible_interp}')
        IV_mesh = griddata((M, T), IV, (M_mesh, T_mesh), method=mesh_config.interp_method)

        # Fit the data to bivariate spline for later usage
        spline = SmoothBivariateSpline(M, T, IV, s=len(M) * 0.01)

        # Use 'nearest' interpolation only for small holes (NaN values) in the vol surface
        nan_mask = np.isnan(IV_mesh)
        if np.any(nan_mask):
            IV_mesh = PlotIV.fill_mesh_nan_with_nearest_interp(M, T, IV, M_mesh, T_mesh, IV_mesh, nan_mask)

        # Mild smoothing (very small), try only if mesh looks noisy
        IV_mesh = gaussian_filter(IV_mesh, sigma=0.6)

        # Fill in mesh output object
        mesh_output = MeshOutput(
            maturity_mesh=T_mesh,
            moneyness_mesh=M_mesh,
            iv_mesh=IV_mesh,
            maturity_grid=T_grid,
            moneyness_grid=M_grid,
            spline_fit=spline
        )

        return mesh_output

    @staticmethod
    def opt_state(moneyness_value: float, opt_type: str, liquidity_flag: bool) -> str:
        """Return a short label describing option moneyness state.

        If `liquidity_flag` is True an empty label is returned. Otherwise the
        method returns one of: ' (ATM)', ' (ITM)' or ' (OTM)' depending on
        `moneyness_value` and `opt_type`.

        Args:
            moneyness_value (float): Strike divided by spot.
            opt_type (str): 'CALL' or 'PUT'.
            liquidity_flag (bool): Suppress labeling when True.

        Returns:
            str: Parenthesized label to append to legend/labels (may be empty).
        """

        # Ignore labeling when liquidity_flag is on
        if liquidity_flag:
            return ''

        # Set ATM if the value is within a threshold of 1.0
        if abs(moneyness_value - 1.0) <= 1e-3:
            return ' (ATM)'

        # Check for wrong opt_type input
        if opt_type.upper() not in ['CALL', 'PUT']:
            raise ValueError(f"The option type {opt_type} is invalid, it can either be CALL or PUT.")

        # Decide whether the option is ATM or ITM according to the PUT and CALL definition
        if opt_type.upper() == 'CALL':
            opt_def_map = {'ITM': (moneyness_value < 1), 'OTM': (moneyness_value > 1)}
        else:
            opt_def_map = {'ITM': (moneyness_value > 1), 'OTM': (moneyness_value < 1)}

        return f" ({next(state for state, is_true in opt_def_map.items() if is_true)})"

    @staticmethod
    def plot_iv_surface(eq_ticker: str,
                        opt_type_label: str,
                        mesh_config: MeshConfig,
                        mesh_output: MeshOutput) -> None:
        """Render and save a 3D plot of the interpolated implied volatility surface.

        The function converts IV values to percentages, masks invalid points,
        draws a matplotlib 3D surface, saves it to 'volatility_surface.png' and
        displays the figure.

        Args:
            eq_ticker (str): Equity ticker used in the title.
            opt_type_label (str): Label describing the option type / liquidity.
            mesh_config (MeshConfig): Mesh configuration (used for rcount/ccount).
            mesh_output (MeshOutput): Mesh data to plot.
        """

        # Interpolated surface (no heavy smoothing)
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Convert to percent for display; mask any remaining NaNs
        plot_mesh = np.ma.masked_invalid(mesh_output.iv_mesh * 100.0)

        # Configure the implied volatility surface plot
        surf = ax.plot_surface(mesh_output.moneyness_mesh,
                               mesh_output.maturity_mesh,
                               plot_mesh,
                               cmap='viridis',
                               rcount=mesh_config.nM,
                               ccount=mesh_config.nT,
                               linewidth=0,
                               antialiased=True,
                               edgecolor='none'
                               )

        # Setup 3D axis, title and legend and POV
        ax.view_init(elev=16, azim=64)
        ax.set_xlabel('Moneyness')
        ax.set_ylabel('Time to Maturity (yrs)')
        ax.set_zlabel('Implied Volatility [%]')
        ax.set_title(
            f"Interpolated IV Surface ({opt_type_label}) - ({eq_ticker})")
        fig.colorbar(surf, shrink=0.5, aspect=10, label='IV [%]')

        # Save figure and show the plot
        plt.tight_layout()
        plt.savefig('volatility_surface.png', dpi=150, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_iv_smile(ax1: Axes,
                      eq_ticker: str,
                      opt_type_label: str,
                      mesh_output: MeshOutput) -> Axes:
        """Plot several cross-section smiles (IV vs moneyness) at selected maturities.

        Args:
            ax1 (matplotlib.axes.Axes): Axis to draw the smile on.
            eq_ticker (str): Equity ticker used in the title.
            opt_type_label (str): Option type label for the title/legend.
            mesh_output (MeshOutput): Mesh and spline fit used to evaluate smiles.

        Returns:
            matplotlib.axes.Axes: The same axis instance passed in (useful for chaining).
        """

        # Get the grid values for moneyness and maturity from the mesh output
        M_grid = mesh_output.moneyness_grid
        T_grid = mesh_output.maturity_grid

        # Configure each curve parameter (maturity and color of each plot)
        plot_config = [
            {'maturity': T_grid[len(T_grid) // 5], 'curve_color': '#2196F3'},
            {'maturity': T_grid[len(T_grid) // 2], 'curve_color': '#FF9800'},
            {'maturity': T_grid[-1], 'curve_color': '#4CAF50'}
        ]

        # Set each plot configuration
        for map_config in plot_config:
            T_level = map_config['maturity']
            iv_smile_curve = mesh_output.spline_fit(M_grid, np.full_like(M_grid, T_level), grid=False)
            ax1.plot(
                M_grid,
                iv_smile_curve,
                color= map_config['curve_color'],
                linewidth=2,
                label=f"T = {T_level:.3f} years"
            )

        # Define axis, title and legend (place an intermittent line in the ATM moneyness 1.0)
        ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='ATM')
        ax1.set_xlabel("Moneyness")
        ax1.set_ylabel("Implied Volatility")
        ax1.set_title(f"Volatility Smile per Maturity ({opt_type_label}) - ({eq_ticker})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        return ax1

    @staticmethod
    def plot_term_structure(ax2: Axes,
                            eq_ticker: str,
                            opt_type: str,
                            opt_type_label: str,
                            mesh_output: MeshOutput,
                            liquidity_flag: bool) -> Axes:
        """Plot implied volatility term-structure for a few representative moneyness levels.

        Args:
            ax2 (matplotlib.axes.Axes): Axis to draw the term-structure on.
            eq_ticker (str): Equity ticker used in the title.
            opt_type (str): 'CALL' or 'PUT' (used to decide ITM/OTM labels).
            opt_type_label (str): Label displayed in the plot title/legend.
            mesh_output (MeshOutput): Mesh and spline fit used to evaluate term curves.
            liquidity_flag (bool): When True, suppresses ITM/OTM labels in the legend.

        Returns:
            matplotlib.axes.Axes: The same axis instance passed in.
        """

        # Gather the grid from the mesh output
        T_grid = mesh_output.maturity_grid

        # Configure each curve parameter (moneyness value, state of the option and color of each plot)
        plot_config = [
            {'moneyness': 0.9,
             'curve_label': f"m = {0.9}{PlotIV.opt_state(0.9, opt_type, liquidity_flag)}",
             'curve_color': '#2196F3'},

            {'moneyness': 1.0,
             'curve_label': f"m = {1.0}{PlotIV.opt_state(1.0, opt_type, liquidity_flag)}",
             'curve_color': '#FF9800'},

            {'moneyness': 1.2,
             'curve_label': f"m = {1.2}{PlotIV.opt_state(1.2, opt_type, liquidity_flag)}",
             'curve_color': '#4CAF50'},
        ]

        # Set each plot configuration
        for map_config in plot_config:
            M_level = map_config['moneyness']
            iv_term_curve = mesh_output.spline_fit(np.full_like(T_grid, M_level), T_grid, grid=False)
            ax2.plot(
                T_grid,
                iv_term_curve,
                color=map_config['curve_color'],
                linewidth=2,
                label=map_config['curve_label']
            )

        # Define axis, title and legend
        ax2.set_xlabel("Maturity (years)")
        ax2.set_ylabel("Implied Volatility")
        ax2.set_title(f"Volatility Term Structure per Moneyness ({opt_type_label}) - ({eq_ticker})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        return ax2


# TODO: Create docstrings and missing comments from the last functions, include this script to github
def main():
    """Example entry point that fetches option data, computes IVs and plots surfaces.

    This function is intended as a demonstration runner: it configures a
    MeshConfig, fetches market option data using `OptMktData.get_option_data`,
    computes implied volatilities with `CalculateIV.calculate_iv`, creates an
    interpolated mesh and spline with `PlotIV.create_mesh_and_interpolate` and
    finally produces a 3D surface and 2D cross-section plots which are saved to
    disk.
    """

    # Input equity ticker and option type
    eq_ticker = 'AAPL'
    opt_type = 'CALL'
    risk_free_rate = 0.05
    liquidity_flag = True

    # Input strike and implied volatility limits
    mesh_config = MeshConfig(
        interp_method='linear',
        min_strike=50.0,
        max_strike=400.0,
        min_iv=0.0,
        max_iv=1.0,
        nM=60,
        nT=40
    )

    # Define label according to liquidity_flag
    opt_type_label = opt_type if not liquidity_flag else "liquid CALL and PUT"

    print("Fetching option market data.")
    df_opt_data = OptMktData.get_option_data(
        eq_ticker,
        risk_free_rate,
        mesh_config,
        opt_type,
        liquidity_flag
    )

    print("Calculating Black-Scholes implied volatility.")
    df_opt_data['implied_volatility'] = df_opt_data.apply(lambda row: CalculateIV.calculate_iv(
        row['opt_mkt_price'],
        row['spot'],
        row['strike'],
        row['risk_free_rate'],
        row['expiry'],
        row['option_type']
    ),
    axis=1)
    # Filter out NaN implied volatility
    df_opt_data = df_opt_data[~df_opt_data['implied_volatility'].isnull()].reset_index(drop=True)

    print("Creating implied volatility surface mesh.")
    mesh_output = PlotIV.create_mesh_and_interpolate(df_opt_data, mesh_config)

    # 3D Plot of vol surface K x T x IV
    print("Plotting implied volatility surface.")
    PlotIV.plot_iv_surface(eq_ticker, opt_type_label, mesh_config, mesh_output)

    # 2D Plots of vol smile (K x IV) and term structure (T x IV)
    print("Plotting implied volatility term structure and smile.")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = PlotIV.plot_iv_smile(ax1, eq_ticker, opt_type_label, mesh_output)
    ax2 = PlotIV.plot_term_structure(ax2, eq_ticker, opt_type, opt_type_label, mesh_output, liquidity_flag)
    plt.tight_layout()
    plt.savefig('cross_sections.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()

    