#src\quantmetrics\price_calculators\ljd_pricing\ljd_characteristic_function.py

import numpy as np
from typing import TYPE_CHECKING

from quantmetrics.risk_calculators.martingale_equation import RiskPremium
from quantmetrics.utils.exceptions import UnsupportedEMMTypeError

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option

class LJDCharacteristicFunction:
    """
    Implements the characteristic function for a lognormal jump-diffusion (LJD) model.

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
        Calculate the characteristic function for the LJD model.

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
        The characteristic function of the LJD under the risk-neutral measure is defined as follows:

        - If ``emm = "mean-correcting"``:

        .. math::

            \\Phi^{\\mathbb{Q}}(u) = \\exp\\left\{T \\left[i u b^\\mathbb{Q} -\\frac{u^2}{2}c + \\lambda \\left( e^{iu \\mu_j - \\frac{u^2}{2}\\sigma_J^2} - 1 \\right) \\right]\\right\},

        where

        .. math::

            b^\\mathbb{Q}= r - \\frac{\\sigma^2}{2} -\\lambda \\kappa \\quad c = \\sigma^2

        .. math::

            \\kappa = \\exp \\left(\\mu_J + \\frac{\\sigma_J^2}{2} \\right) - 1

        - If ``emm = "Esscher"``:

        .. math::

            \\Phi^{\\mathbb{Q}}(u) = \\exp\\left\{T \\left[i u b^\\mathbb{Q} -\\frac{u^2}{2}c + \\lambda^\\mathbb{Q} \\left( \\left[e^{iu \\mu_j - \\frac{u^2}{2}\\sigma_J^2} e^{iu \\theta \\sigma_J^2}\\right]^{\\frac{1}{g(\\psi)}} - 1 \\right) \\right]\\right\},

        where
        
        .. math::

            b^\\mathbb{Q}= r - \\frac{\\sigma^2}{2} -\\lambda \\kappa + \\theta(\\psi) \\sigma^2  \\quad c = \\sigma^2

        .. math::

            \\lambda^\\mathbb{Q} = \\lambda f(\\theta)

        .. math::

            f(y) = \\frac{1}{\\sqrt{g(\\psi)}} \\exp \\left[\\frac{1}{g(\\psi)} \\left(\\mu_J y + \\frac{\\sigma_J^2}{2} y^2 + \\psi \\mu_J^2  \\right)  \\right] \\quad \\text{and} \\quad g(\\psi) = 1 - 2\\psi \\sigma_J^2

        with

        .. math::

            \\psi < \\frac{1}{2\\sigma_J^2}    

        The first-order Esscher parameter :math:`\\theta` is the risk premium (market price of risk) and which is the unique solution to the martingale equation for each :math:`\\psi` which is the second-order Esscher parameter. See the documentation of the ``RiskPremium`` class for the martingale equation and refer to [1]_ for more details.
        
        - :math:`\\mathbb{Q}` is the risk-neutral measure.
        - :math:`T` is the time to maturity.
        - :math:`i` is the imaginary unit.
        - :math:`u` is the input variable.
        - :math:`r` is the risk-free interest rate.
        - :math:`\\sigma` is the volatility of the underlying asset.
        - :math:`\\mu_J` is the mean of the jump sizes.
        - :math:`\\sigma_J` is the standard deviation of the jump sizes.
        - :math:`\\lambda` is the jump intensity rate.   

        References
        ----------

        .. [1] Choulli, T., Elazkany, E., & Vanmaele, M. (2024). Applications of the Second-Order Esscher Pricing in Risk Management. arXiv preprint arXiv:2410.21649.
        
        .. [2] Matsuda, K. (2004). Introduction to option pricing with Fourier transform: Option pricing with exponential Lévy models. Department of Economics The Graduate Center, The City University of New York, 1-241.

        .. [3] Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. Journal of financial economics, 3(1-2), 125-144.
        """
        return self._ljd_characteristic_function(u)

    def _ljd_characteristic_function(self, u: np.ndarray) -> np.ndarray:
        """
        Calculate the characteristic function for the LJD model.

        Parameters
        ----------
        u : np.ndarray
            Input array for the characteristic function.

        Returns
        -------
        np.ndarray
            The characteristic function values.
        """
        mu = self.model.mu
        sigma = self.model.sigma
        lambda_ = self.model.lambda_
        muJ = self.model.muJ
        sigmaJ = self.model.sigmaJ
        r = self.option.r
        T = self.option.T
        emm = self.option.emm
        psi = self.option.psi

        if emm == "mean-correcting":
            b = r - sigma**2 / 2 - lambda_ * (np.exp(muJ + sigmaJ**2 / 2) - 1)
            char_func = np.exp(
                T
                * (
                    1j * u * b
                    - sigma**2 * u**2 / 2
                    + lambda_ * (np.exp(1j * u * muJ - u**2 * sigmaJ**2 / 2) - 1)
                )
            )
        elif emm == "Esscher":
            theta = RiskPremium(self.model, self.option).calculate()
            b = (
                mu
                - 0.5 * sigma**2
                - lambda_ * (np.exp(muJ + sigmaJ**2 / 2) - 1)
                + theta * sigma**2
            )

            g_psi = 1 - 2 * psi * sigmaJ**2

            f = lambda x: np.exp(
                (muJ * x + 0.5 * sigmaJ**2 * x**2 + psi * muJ**2) / g_psi
            ) / (g_psi**0.5)

            char_func = np.exp(
                T
                * (
                    1j * u * b
                    - u**2 * sigma**2 / 2
                    + lambda_ * (f(theta + 1j * u) - f(theta))
                )
            )
        else:
            raise UnsupportedEMMTypeError(emm)

        return char_func