import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from functools import partial

from quantmetrics.levy_models import GBM
from quantmetrics.option_pricing import Option, OptionPricer


class BlackScholesImpliedVolatility:
    """
    A class for calculating the implied volatility given option prices.

    Parameters
    ----------
    initial_price : float
        Initial underlying price.
    time_to_maturity : float
        Time to maturity of the option (annualized).
    interest_rate : float
        Interest rate (annualized).
    strike_prices : np.ndarray
        An array of strike prices.
    option_prices : np.ndarray
        An array of option prices with the same dimension as strike_prices. This can be either the market option prices or option prices calculated from a model after calibration.
    initial_sigma : float
        Initial estimate of the volatility.

    Returns
    -------
    np.ndarray
        An array of the Black-Scholes implied volatilities with same size as K an option_prices

    References
    ----------
    Hilpisch, Y. (2015). Derivatives analytics with Python: data analysis, models, simulation, calibration and hedging. John Wiley & Sons.

    Examples
    --------
    Example of usage:

    ```python
    import numpy as np
    from quantmetrics.implied_volatility import ImpliedVolatility

    S0 = 100
    T = 1.0
    r = 0.05
    K = np.array([100, 110, 120])
    option_prices = np.array([10, 8, 6])
    init_sigma = 0.2

    iv_calculator = ImpliedVolatility(S0, T, r, K, option_prices, init_sigma)
    implied_vols = iv_calculator.calculate_iv()
    print("Implied Volatilities:", implied_vols)
    ```
    """

    def __init__(
        self,
        initial_price: float,
        time_to_maturity: float,
        interest_rate: float,
        strike_prices: np.ndarray,
        option_prices: np.ndarray,
        initial_sigma: float = 0.2,
    ):
        self.initial_price = initial_price
        self.time_to_maturity = time_to_maturity
        self.interest_rate = interest_rate
        self.strike_prices = strike_prices
        self.option_prices = option_prices
        self.initial_sigma = initial_sigma

    def _error(self, sigma: float, strike_price: float, option_price: float) -> float:
        """
        An error function which returns the difference between option prices.

        Parameters
        ----------
        sigma : float
            Volatility estimate.
        strike_price : float
            Strike price of the option.
        option_price : float
            Option price (market option prices or model option prices).

        Returns
        -------
        float
            Difference between the given option price and the Black-Scholes option price.
        """
        gbm = GBM(S0=self.initial_price, sigma=sigma)
        option = Option(T=self.time_to_maturity, r=self.interest_rate, K=strike_price)
        gbm_price = OptionPricer(gbm, option)

        return gbm_price.fft() - option_price

    def calculate_iv(self) -> np.ndarray:
        """
        Calculate the implied volatility for each option price.

        Returns
        -------
        np.ndarray
            An array of implied volatilities corresponding to the strike prices and option prices.
        """
        implied_vols = np.zeros_like(self.strike_prices, dtype=float)

        for i, (strike_price, option_price) in enumerate(
            zip(self.strike_prices, self.option_prices)
        ):
            # Use functools.partial to wrap the _error function
            error_func = partial(
                self._error, strike_price=strike_price, option_price=option_price
            )
            implied_vols[i] = fsolve(error_func, self.initial_sigma)[0]

        return implied_vols

    def plot_iv_vs_strike(self):
        """
        Plot the implied volatilities against the strike prices.
        """
        implied_vols = self.calculate_iv()

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(self.strike_prices, implied_vols, marker="o", linestyle="-", color="b")
        plt.xlabel("Strike Price")
        plt.ylabel("Implied Volatility")
        plt.title("Implied Volatility vs Strike Price")
        plt.grid(True)
        plt.show()
