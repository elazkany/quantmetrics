# option_pricing/exact_equation.py

from quantmetrics.levy_models import GBM

from typing import TYPE_CHECKING
import numpy as np
import scipy.stats as st

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option


class ExactSolution:
    def __init__(
        self,
        model: "LevyModel",
        option: "Option",
    ):
        """
        Initialize the ExactSolution with a model and an option.

        Parameters
        ----------
        model : LevyModel
            A Levy model used for pricing the option.
        option : Option
            The option parameters including interest rate, strike price, etc.
        """
        self.model = model
        self.option = option

    def calculate(self) -> float:
        """
        Calculate the option price using the exact solution.

        Returns
        -------
        float
            The calculated option price.
        """
        if isinstance(self.model, GBM):
            return self._black_scholes_exact_price()

    def _black_scholes_exact_price(self):
        """
        Calculate the European option price using the Black-Scholes exact equation.

        Returns
        -------
        float
            The calculated option price.
        """
        S0 = self.model.S0
        sigma = self.model.sigma
        r = self.option.r
        q = self.option.q
        K = self.option.K
        T = self.option.T
        payoff = self.option.payoff

        # Calculate d_plus and d_minus
        d_plus = (np.log(S0 / K) + (r - q + sigma**2 / 2) * T) / (sigma * T**0.5)

        d_minus = d_plus - sigma * T**0.5

        # Calculate the option price based on the payoff type
        if payoff == "c":
            option_price = np.exp(-q * T) * S0 * st.norm.cdf(d_plus) - K * np.exp(
                -r * T
            ) * st.norm.cdf(d_minus)
        else:
            option_price = K * np.exp(-r * T) * st.norm.cdf(
                -d_minus
            ) - S0 * st.norm.cdf(-d_plus)
        return option_price
