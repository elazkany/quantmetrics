#src\quantmetrics\price_calculators\gbm_pricing\gbm_closed_form.py

import numpy as np
import scipy.stats as st
from typing import TYPE_CHECKING, Union

from quantmetrics.utils.exceptions import UnknownPayoffTypeError

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option

class GBMClosedForm:
    """
        Implements the closed-form solution for pricing European options under a Geometric Brownian Motion (GBM) model.

        Parameters
        ----------
        model : LevyModel
            A LevyModel object specifying the underlying asset's model and its parameters.
        option : Option
            An Option object specifying the option parameters: interest rate, strike price, time to maturity, dividend yield and the equivalent martingale measure.
        """
    def __init__(self, model: "LevyModel", option: "Option"):
        self.model = model
        self.option = option

    def calculate(self) -> Union[float, np.ndarray]:
        """
        Calculate the European option price using the Black-Scholes exact equation.

        Returns
        -------
        Union[float, np.ndarray]
            The calculated option price.

        Notes
        -------
        The closed-form solution for the European call option:

        .. math::

            C = e^{-qT}S_0 \\ N(d_+) - e^{-rT}K \\ N(d_-)

        and for the European put option:

        .. math::

            P = e^{-rT}K\\ N(-d_-) - e^{-qT} S_0 \\ N(-d_+)

        Where:

            .. math::
            
                d_+ = \\frac{\\ln(\\frac{e^{-qT}S_0}{K}) + (r + \\frac{\\sigma^2}{2})T}{\\sigma \\sqrt{T}} \\quad \\text{and} \\quad d_- = d_+ - \\sigma \\sqrt{T}

            - :math:`C` is the call option price.
            - :math:`P` is the put option price.
            - :math:`q` is the dividend yield.
            - :math:`T` is the time to maturity.
            - :math:`S_0` is the underlying price.
            - :math:`r` is the risk-free interest rate.            
            - :math:`K` is the strike price.
            - :math:`\\sigma` is the volatility.
            - :math:`N(x)` is the standard normal cumulative distribution function: :math:`N(x)= \\frac{1}{\\sqrt{2\pi}}\\int_{-\\infty}^x e^{-\\frac{u^2}{2}}du`.
        

        Examples
        --------
        >>> from quantmetrics.levy_models import GBM
        >>> from quantmetrics.option_pricing import Option, OptionPricer
        >>> gbm = GBM() # S0=50, sigma=0.2
        >>> option = Option(K=np.array([20,50,80])) # r=0.05, q=0.02, T=0.5
        >>> gbm_pricer = OptionPricer(gbm, option)
        >>> gbm_pricer.closed_form()
        array([2.99962934e+01, 3.15381758e+00, 1.52305781e-03])

        References
        ----------

        .. [1] Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of political economy, 81(3), 637-654.
        
        .. [2] Matsuda, K. (2004). Introduction to option pricing with Fourier transform: Option pricing with exponential Lévy models. Department of Economics The Graduate Center, The City University of New York, 1-241.
        """
        return self._closed_form_solution()
    
    def _closed_form_solution(self) -> Union[float, np.ndarray]:
        #Implement GBM closed form price calculations
        S0 = self.model.S0
        sigma = self.model.sigma
        r = self.option.r
        q = self.option.q
        K = self.option.K
        T = self.option.T
        payoff = self.option.payoff

        # Calculate d_plus and d_minus
        d_plus = (np.log(np.exp(-q * T) * S0 / K) + (r + sigma**2 / 2) * T) / (sigma * T**0.5)

        d_minus = d_plus - sigma * T**0.5

        # Calculate the option price based on the payoff type
        if payoff == "c":
            option_price = np.exp(-q * T) * S0 * st.norm.cdf(d_plus) - K * np.exp(
                -r * T
            ) * st.norm.cdf(d_minus)
        elif payoff == "p":
            option_price = K * np.exp(-r * T) * st.norm.cdf(
                -d_minus
            ) - np.exp(-q * T) * S0 * st.norm.cdf(-d_plus)
        else:
            raise UnknownPayoffTypeError(payoff)
        return option_price