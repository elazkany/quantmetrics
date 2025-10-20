#src\quantmetrics\price_calculators\cjd_pricing\cjd_characteristic_function.py

import numpy as np
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option

class CJDCharacteristicFunction:
    """
    Implements the characteristic function for a constant jump-diffusion (CJD) model.

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
            exact = True,
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
        Calculate the characteristic function for the CJD model.

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
        The characteristic function of the CJD under the risk-neutral measure is defined as follows:

        - If ``emm = "mean-correcting"``:

        .. math::

            \\Phi^{\\mathbb{Q}}(u) = \\exp\\left\{T \\left[i u b^\\mathbb{Q} -\\frac{u^2}{2}c + \\lambda \\left( e^{iu \\gamma } - 1 \\right) \\right]\\right\},

        where

        .. math::

            b^\\mathbb{Q}= r - \\frac{\\sigma^2}{2} -\\lambda \\kappa \\quad c = \\sigma^2

        .. math::

            \\kappa = e^\\gamma - 1

        - If ``emm = "Esscher"``:

        .. math::

            \\Phi^{\\mathbb{Q}}(u) = \\exp\\left\{T \\left[i u b^\\mathbb{Q} -\\frac{u^2}{2}c + \\lambda^\\mathbb{Q} \\left( e^{iu\\gamma} - 1 \\right) \\right]\\right\},

        where
        
        .. math::

            b^\\mathbb{Q}= \\mu - \\frac{\\sigma^2}{2} -\\lambda \\kappa + \\theta \\sigma^2  \\quad c = \\sigma^2

        .. math::

            \\lambda^\\mathbb{Q} = \\lambda \\exp \\left(\\theta \\gamma+ \\psi \\gamma^2 \\right)
  
        The first-order Esscher parameter :math:`\\theta` is the risk premium (market price of risk) and which is the unique solution to the martingale equation for each :math:`\\psi` which is the second-order Esscher parameter. See the documentation of the ``RiskPremium`` class for the martingale equation and refer to [1]_ for more details.
        
        - :math:`\\mathbb{Q}` is the risk-neutral measure.
        - :math:`T` is the time to maturity.
        - :math:`i` is the imaginary unit.
        - :math:`u` is the input variable.
        - :math:`r` is the risk-free interest rate.
        - :math:`\\sigma` is the volatility of the underlying asset.
        - :math:`\\gamma` is the constant jump size.
        - :math:`\\lambda` is the jump intensity rate. 

        References
        ----------

        .. [1] Choulli, T., Elazkany, E., & Vanmaele, M. (2024). Applications of the Second-Order Esscher Pricing in Risk Management. arXiv preprint arXiv:2410.21649.
        
        .. [2] Matsuda, K. (2004). Introduction to option pricing with Fourier transform: Option pricing with exponential Lévy models. Department of Economics The Graduate Center, The City University of New York, 1-241.

        .. [3] Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. Journal of financial economics, 3(1-2), 125-144.
        """
        return self._cjd_characteristic_function(
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

    def _cjd_characteristic_function(
            self,
            u: np.ndarray,
            exact = True,
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
        Calculate the characteristic function for the CJD model.

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
        gamma = self.model._gamma
        r = self.option.r
        q = self.option.q
        T = self.option.T
        emm = self.option.emm
        psi = self.option.psi

        gamma_tilde = np.exp(gamma) - 1

        if emm == "mean-correcting":
            b = r - sigma*sigma / 2 - lambda_ * gamma_tilde
            return np.exp(
                T
                * (
                    1j * u * b
                    - sigma*sigma * u*u / 2
                    + lambda_ * (np.exp(1j * u * gamma) - 1)
                )
            )
        elif emm == "Esscher":
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

            b = mu - sigma*sigma / 2 - lambda_ * gamma_tilde + theta * sigma*sigma
            def f(theta):
                exp0 = theta * gamma + psi * gamma * gamma
                return np.exp(np.clip(exp0, -EXP_CLIP, EXP_CLIP))
            
            return np.exp(
                T
                * (
                    1j * u * b
                    - u*u * sigma*sigma / 2
                    + lambda_ * (f(theta + 1j * u)- f(theta))
                )
            )
        else:
            raise ValueError(f"Unknown or unsupported EMM type: {emm}. emm is either 'mean-correcting' or 'Esscher'.")
