# quantmetrics/option_pricing/martingale_equation.py

from quantmetrics.levy_models import GBM, CJD, LJD

from typing import TYPE_CHECKING
import numpy as np
import scipy.optimize

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option


class RiskPremium:
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
        if isinstance(self.model, GBM):
            return self._black_scholes_risk_premium()
        elif isinstance(self.model, CJD) and self.option.emm == "Black-Scholes":
            return self._black_scholes_risk_premium()
        elif isinstance(self.model, CJD) and self.option.emm == "Esscher":
            return self._cjd_esscher_risk_premium()
        elif isinstance(self.model, LJD) and self.option.emm == "Black-Scholes":
            return self._black_scholes_risk_premium()
        elif isinstance(self.model, LJD) and self.option.emm == "Esscher":
            return self._ljd_esscher_risk_premium()
        else:
            print("No matching condition found.")

    def _black_scholes_risk_premium(self):
        mu = self.model.mu
        sigma = self.model.sigma
        r = self.option.r
        # definition of the risk-premium
        theta = (r - mu) / (sigma * sigma)
        return theta

    def _cjd_esscher_risk_premium(self):
        mu = self.model.mu
        sigma = self.model.sigma
        lambda_ = self.model.lambda_
        gamma = self.model.gamma
        r = self.option.r
        psi = self.option.psi

        gamma_tilde = np.exp(gamma) - 1

        def _martingale_equation(x, psi, mu, sigma, lambda_, gamma, r):
            Exp = np.exp(x * gamma + psi * gamma**2)
            # Exp = np.exp(709)  the max value python can handle
            mc = -r + mu + x * sigma**2 + lambda_ * gamma_tilde * (Exp - 1)
            # if upperLimitCond >= 0:
            # raise ValueError("theoretical condition is invalid")
            return mc

        def _root(psi, mu, sigma, lambda_, gamma, r):
            interval = [-(10 ** (10)), 10000]
            try:
                root = scipy.optimize.brentq(
                    _martingale_equation,
                    interval[0],
                    interval[1],
                    args=(psi, mu, sigma, lambda_, gamma, r),
                )
            except ValueError:
                root = 9999
            return root

        # the Esscher parameter (risk premium)
        theta = _root(
            psi=psi, mu=mu, sigma=sigma, lambda_=lambda_, gamma=gamma, r=r
        )  # * gamma_tilde

        return theta

    def _ljd_esscher_risk_premium(self):
        mu = self.model.mu
        sigma = self.model.sigma
        lambda_ = self.model.lambda_
        muJ = self.model.muJ
        sigmaJ = self.model.sigmaJ
        r = self.option.r
        psi = self.option.psi

        g_psi = 1 - 2 * sigmaJ**2 * psi

        f = lambda x: np.exp(
            (muJ * x + 0.5 * sigmaJ**2 * x**2 + psi * muJ**2) / g_psi
        ) / (g_psi**0.5)

        def _martingale_equation(x, psi, mu, sigma, lambda_, muJ, sigmaJ, r):
            xi = np.exp(muJ + sigmaJ**2 / 2)
            # Exp = np.exp(709)  the max value python can handle
            mc = -r + mu + x * sigma**2 + lambda_ * (f(x + 1) - f(x) - xi + 1)
            # if upperLimitCond >= 0:
            # raise ValueError("theoretical condition is invalid")
            return mc

        def _root(psi, mu, sigma, lambda_, muJ, sigmaJ, r):
            interval = [-150, 150]
            try:
                root = scipy.optimize.brentq(
                    _martingale_equation,
                    interval[0],
                    interval[1],
                    args=(psi, mu, sigma, lambda_, muJ, sigmaJ, r),
                )
            except ValueError:
                root = 9999
            return root

        # the Esscher parameter (risk premium)
        theta = _root(
            psi=psi, mu=mu, sigma=sigma, lambda_=lambda_, muJ=muJ, sigmaJ=sigmaJ, r=r
        )

        return theta
