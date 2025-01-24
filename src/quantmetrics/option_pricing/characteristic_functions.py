# option_pricing/characteristics_functions.py

from quantmetrics.levy_models import GBM, CJD, LJD, DEJD
from quantmetrics.option_pricing import RiskPremium

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option


class CharacteristicFunction:
    def __init__(self, model: "LevyModel", option: "Option"):
        """
        Initialize the CharacteristicFunction with a model and an option.

        Parameters
        ----------
        model : LevyModel
            The Levy model used for calculating the characteristic function.
        option : Option
            The option parameters including interest rate, volatility, etc.
        """
        self.model = model
        self.option = option

    def calculate(self, u: np.ndarray) -> np.ndarray:
        """
        Calculate the characteristic function for the given model.

        Parameters
        ----------
        u : np.ndarray
            Input array for the characteristic function.

        Returns
        -------
        np.ndarray
            The characteristic function values.
        """
        if isinstance(self.model, GBM):
            return self._gbm_characteristic_function(u)
        elif isinstance(self.model, CJD):
            return self._cjd_characteristic_function(u)
        elif isinstance(self.model, LJD):
            return self._ljd_characteristic_function(u)
        elif isinstance(self.model, DEJD):
            return self._dejd_characteristic_function(u)
        else:
            pass

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

    def _cjd_characteristic_function(self, u: np.ndarray) -> np.ndarray:
        mu = self.model.mu
        sigma = self.model.sigma
        lambda_ = self.model.lambda_
        gamma = self.model.gamma
        r = self.option.r
        q = self.option.q
        K = self.option.K
        T = self.option.T
        payoff = self.option.payoff
        emm = self.option.emm
        psi = self.option.psi

        gamma_tilde = np.exp(gamma) - 1

        if emm == "Black-Scholes":
            b = r - sigma**2 / 2 - lambda_ * gamma_tilde
            char_func = np.exp(
                T
                * (
                    1j * u * b
                    - sigma**2 * u**2 / 2
                    + lambda_ * (np.exp(1j * u * gamma) - 1)
                )
            )
        else:
            theta = RiskPremium(self.model, self.option).calculate()
            b = mu - sigma**2 / 2 - lambda_ * gamma_tilde + theta * sigma**2
            char_func = np.exp(
                T
                * (
                    1j * u * b
                    - u**2 * sigma**2 / 2
                    + lambda_
                    * (
                        np.exp((theta + 1j * u) * gamma + psi * gamma**2)
                        - np.exp(theta * gamma + psi * gamma**2)
                    )
                )
            )

        return char_func

    def _ljd_characteristic_function(self, u: np.ndarray) -> np.ndarray:
        mu = self.model.mu
        sigma = self.model.sigma
        lambda_ = self.model.lambda_
        muJ = self.model.muJ
        sigmaJ = self.model.sigmaJ
        r = self.option.r
        q = self.option.q
        K = self.option.K
        T = self.option.T
        payoff = self.option.payoff
        emm = self.option.emm
        psi = self.option.psi

        if emm == "Black-Scholes":
            b = r - sigma**2 / 2 - lambda_ * (np.exp(muJ + sigmaJ**2 / 2) - 1)
            char_func = np.exp(
                T
                * (
                    1j * u * b
                    - sigma**2 * u**2 / 2
                    + lambda_ * (np.exp(1j * u * muJ - u**2 * sigmaJ**2 / 2) - 1)
                )
            )
        else:
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

        return char_func

    def _dejd_characteristic_function(self, u: np.ndarray) -> np.ndarray:
        mu = self.model.mu
        sigma = self.model.sigma
        lambda_ = self.model.lambda_
        eta1 = self.model.eta1
        eta2 = self.model.eta2
        p = self.model.p
        r = self.option.r
        q = self.option.q
        K = self.option.K
        T = self.option.T
        payoff = self.option.payoff
        emm = self.option.emm
        psi = self.option.psi

        if emm == "Black-Scholes":
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
            theta = RiskPremium(self.model, self.option).calculate()

            b = (
                mu
                - 0.5 * sigma**2
                - lambda_ * (p / (eta1 - 1) - (1 - p) / (eta2 + 1) + theta * sigma**2)
            )

            char_func = np.exp(
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
        else:
            pass

        return char_func
