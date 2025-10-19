#src\quantmetrics\price_calculators\dejd_pricing\dejd_characteristic_function.py

import numpy as np
from typing import TYPE_CHECKING

from quantmetrics.risk_calculators.martingale_equation import RiskPremium
from quantmetrics.utils.exceptions import FeatureNotImplementedError, UnsupportedEMMTypeError

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option

class DEJDCharacteristicFunction:
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

    def __call__(
            self,
            u: np.ndarray,
            exact = False,
            theta = None,
            L=1e-12,
            M=1.0,
            N_center=150,
            N_tails=100,
            EXP_CLIP=700,
            search_bounds = (-50, 50),
            xtol=1e-8,
            rtol=1e-8,
            maxiter=500,
            M_int = 100,
            N_int = 10_000,
            sanity_theta: float = 1.0,
            chunk_u = None,
            ) -> np.ndarray:
        """
        Calculate the characteristic function for the DEJD model.

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
        The characteristic function of the DEJD under the risk-neutral measure is defined as follows:

        - If ``emm = "mean-correcting"``:

        .. math::

            \\Phi^{\\mathbb{Q}}(u) = \\exp\\left\{T \\left[i u b^\\mathbb{Q} -\\frac{u^2}{2}c + \\lambda \\left( \\frac{p\\eta_1}{\\eta_1 - iu} + \\frac{q \\eta_2}{\\eta_2 + iu} - 1 \\right) \\right]\\right\},

        where

        .. math::

            b^\\mathbb{Q}= r - \\frac{\\sigma^2}{2} -\\lambda \\kappa \\quad c = \\sigma^2

        .. math::

            \\kappa = \\frac{p\\eta_1}{\\eta_1 -1} + \\frac{q\\eta_2}{\\eta_2+1} - 1, \\quad q = 1-p

        - If ``emm = "Esscher"`` and ``psi = 0``:

        .. math::

            \\Phi^{\\mathbb{Q}}(u) = \\exp\\left\{T \\left[i u b^\\mathbb{Q} -\\frac{u^2}{2}c + \\lambda^\\mathbb{Q} \\left( \\frac{\\frac{p\\eta_1}{\\eta_1 - (\\theta +iu)} + \\frac{q\\eta_2}{\\eta_2 + (\\theta + iu)}}{\\frac{p\\eta_1}{\\eta_1 - \\theta} + \\frac{q\\eta_2}{\\eta_2 + \\theta}} - 1 \\right) \\right]\\right\},

        where
        
        .. math::

            b^\\mathbb{Q}= r - \\frac{\\sigma^2}{2} -\\lambda \\kappa + \\theta \\sigma^2  \\quad c = \\sigma^2

        .. math::

            \\lambda^\\mathbb{Q} = \\lambda \\left( \\frac{p\\eta_1}{\\eta_1 - \\theta} + \\frac{q\\eta_2}{\\eta_2 + \\theta} \\right)

        - If ``emm = "Esscher"`` and ``psi != 0``:

        .. math::

            \\Phi^{\\mathbb{Q}}(u) = \\exp\\left\{T \\left[i u b^\\mathbb{Q} -\\frac{u^2}{2}c + \\lambda^\\mathbb{Q} \\left(\\frac{p\\eta_1 I(\\theta+iu, \\eta_1, \\psi) + q\\eta_2 I(\\theta +iu, -\\eta_2, \\psi)}{p\\eta_1 I(\\theta, \\eta_1, \\psi) + q\\eta_2 I(\\theta, -\\eta_2, \\psi)} - 1 \\right)\\right] \\right\},

        where
        
        .. math::

            b^\\mathbb{Q}= r - \\frac{\\sigma^2}{2} -\\lambda \\kappa + \\theta(\\psi) \\sigma^2  \\quad c = \\sigma^2

        .. math::

            \\lambda^\\mathbb{Q} = \\lambda \\frac{1}{2}\\sqrt{\\frac{\\pi}{|\\psi|}} \\left[p\\eta_1 I(\\theta, \\eta_1, \\psi) + q\\eta_2 I(\\theta, -\\eta_2, \\psi) \\right],

        .. math::

            I(a,b,\\psi):= \\exp \\left[-\\psi \\left(\\frac{a-b}{2\\psi} \\right)^2 \\right] \\left\{1 - erf\\left[\\sqrt{|\\psi|} \\left(\\frac{a-b}{2\\psi} \\right) \\right]  \\right\}
            
        with

        .. math::

            \\psi < 0.
        
        The first-order Esscher parameter :math:`\\theta` is the risk premium (market price of risk) and which is the unique solution to the martingale equation for each :math:`\\psi` which is the second-order Esscher parameter. See the documentation of the ``RiskPremium`` class for the martingale equation and refer to [1]_ for more details.
        
        - :math:`\\mathbb{Q}` is the risk-neutral measure.
        - :math:`T` is the time to maturity.
        - :math:`i` is the imaginary unit.
        - :math:`u` is the input variable.
        - :math:`r` is the risk-free interest rate.
        - :math:`\\sigma` is the volatility of the underlying asset.
        - :math:`p,q\\geq 0, p+q =1` are the probabilities of upward and downward jumps.
        - The upward jump sizes are exponentially distributed with mean :math:`1/\\eta_1` with :math:`\\eta_1>1`.
        - The downward jump sizes are exponentially distributed with mean :math:`1/\\eta_2` with :math:`\\eta_2 >0`.
        - :math:`\\lambda` is the jump intensity rate.   

        References
        ----------

        .. [1] Choulli, T., Elazkany, E., & Vanmaele, M. (2024). Applications of the Second-Order Esscher Pricing in Risk Management. arXiv preprint arXiv:2410.21649.

        .. [2] Jeanblanc, M., Yor, M., & Chesney, M. (2009). Mathematical methods for financial markets. Springer Science & Business Media.

        .. [3] Kou, S. G. (2002). A jump-diffusion model for option pricing. Management science, 48(8), 1086-1101.
        """
        return self._dejd_characteristic_function(
            u=u,
            exact=exact,
            theta=theta,
            L=L,
            M=M,
            N_center=N_center,
            N_tails=N_tails,
            EXP_CLIP=EXP_CLIP,
            search_bounds=search_bounds,
            xtol=xtol,
            rtol=rtol,
            maxiter=maxiter,
            M_int=M_int,
            N_int=N_int,
            sanity_theta=sanity_theta,
            chunk_u = chunk_u,
        )

    def _dejd_characteristic_function(
            self,
            u: np.ndarray,
            exact = False,
            theta = None,
            L=1e-12,
            M=1.0,
            N_center=150,
            N_tails=100,
            EXP_CLIP=700,
            search_bounds = (-50, 50),
            xtol=1e-8,
            rtol=1e-8,
            maxiter=500,
            sanity_theta: float = 1.0,
            M_int = 100,
            N_int = 10_000,
            chunk_u = None,
    ) -> np.ndarray:
        """
        Calculate the characteristic function for the DEJD model.

        Parameters
        ----------
        u : np.ndarray
            Input array for the characteristic function.

        Returns
        -------
        np.ndarray
            The characteristic function values.
        """
        mu = self.model._mu
        sigma = self.model._sigma
        lambda_ = self.model._lambda_
        eta1 = self.model._eta1
        eta2 = self.model._eta2
        p = self.model._p
        r = self.option.r
        T = self.option.T
        emm = self.option.emm
        psi = self.option.psi

        if emm == "mean-correcting":
            b = (
                r
                - sigma**2 / 2
                - lambda_ * (p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1)
            )
            char_func = np.exp(
                T
                * (
                    1j * u * b
                    - sigma**2 * u**2 / 2
                    + lambda_
                    * (
                        p * eta1 / (eta1 - 1j * u)
                        + (1 - p) * eta2 / (eta2 + 1j * u)
                        - 1
                    )
                )
            )
        elif (emm == "Esscher") & (psi == 0.0):
            if theta is None:
                from quantmetrics.risk_neutral.market_price_of_risk import MarketPriceOfRisk
                theta = MarketPriceOfRisk(self.model, self.option).solve(
                    exact=exact,
                    L=L,
                    M=M,
                    N_center=N_center,
                    N_tails=N_tails,
                    EXP_CLIP=EXP_CLIP,
                    search_bounds=search_bounds,
                    xtol=xtol,
                    rtol=rtol,
                    maxiter=maxiter,
                    sanity_theta=sanity_theta
                )

            b = (
                mu
                - 0.5 * sigma**2
                - lambda_ * (p / (eta1 - 1) - (1 - p) / (eta2 + 1) + theta * sigma**2)
            )

            return np.exp(
                T
                * (
                    1j * u * b
                    - 0.5 * sigma**2 * u**2
                    + lambda_
                    * (
                        (
                            p * eta1 / (eta1 - (theta + 1j * u))
                            + (1 - p) * eta2 / (eta2 + (theta + 1j * u))
                        )
                        - (p * eta1 / (eta1 - theta) + (1 - p) * eta2 / (eta2 + theta))
                    )
                )
            )
        elif (emm == "Esscher") & (psi != 0.0):
            raise FeatureNotImplementedError("Esscher")
        else:
            raise UnsupportedEMMTypeError(emm)

        return char_func