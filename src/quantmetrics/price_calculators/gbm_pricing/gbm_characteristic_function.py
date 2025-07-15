#src\quantmetrics\price_calculators\gbm_pricing\gbm_characteristic_function.py

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option

class GBMCharacteristicFunction:
    """
    Implements the characteristic function for a Geometric Brownian Motion (GBM) model.

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

    def calculate(self, u: np.ndarray) -> np.ndarray:
        """
        Calculate the characteristic function for the GBM model.

        Parameters
        ----------
        u : np.ndarray
            Input array for the characteristic function.

        Returns
        -------
        np.ndarray
            The characteristic function values.

        Notes
        -------
        The characteristic function of the GBM under the risk-neutral measure is defined as follows:

        .. math::

            \\Phi^{\\mathbb{Q}}(u) = \\exp\\left\{T \\left[i u b^\\mathbb{Q} -\\frac{u^2}{2} c \\right]\\right\},

        Where:

        .. math::

            b^{\\mathbb{Q}} = r - \\frac{\\sigma^2}{2}, \\quad c = \\sigma^2

            
        - :math:`\\mathbb{Q}` is the risk-neutral measure.
        - :math:`T` is the time to maturity.
        - :math:`i` is the imaginary unit.
        - :math:`u` is the input variable.
        - :math:`r` is the risk-free interest rate.
        - :math:`\\sigma` is the volatility of the underlying asset.            

        References
        ----------

        .. [1] Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of political economy, 81(3), 637-654.
        
        .. [2] Matsuda, K. (2004). Introduction to option pricing with Fourier transform: Option pricing with exponential Lévy models. Department of Economics The Graduate Center, The City University of New York, 1-241.
        """
        return self._gbm_characteristic_function(u)

    def _gbm_characteristic_function(self, u: np.ndarray) -> np.ndarray:
        """
        Calculate the characteristic function for the GBM model.

        Parameters
        ----------
        u : np.ndarray
            Input array for the characteristic function.

        Returns
        -------
        np.ndarray
            The characteristic function values.
        """
        sigma = self.model.sigma
        r = self.option.r
        T = self.option.T

        b = r - 0.5 * sigma**2
        char_func = np.exp(T * (-0.5 * sigma**2 * u**2 + 1j * u * b))
        return char_func