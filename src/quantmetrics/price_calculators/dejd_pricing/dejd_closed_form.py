#src\quantmetrics\price_calculators\dejd_pricing\dejd_closed_form.py

import numpy as np
import scipy.stats as st
import math
from scipy.special import comb
import scipy.integrate as integrate
from typing import TYPE_CHECKING, Union

from quantmetrics.utils.exceptions import FeatureNotImplementedError, UnsupportedEMMTypeError, UnknownPayoffTypeError

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option

class DEJDClosedForm:
    """
    Implements the closed-form solution for pricing European options under a lognormal jump-diffusion (LJD) model.

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
        Calculate the European option price using the lognormal jump-diffusion exact equation.

        Returns
        -------
        Union[float, np.ndarray]
            The calculated option price.
        
        Notes
        -------
        The closed-form solution for the European option has the following formula"

        - For ``emm = "mean-correcting"``:

        .. math::

            C = S_0 \\Upsilon \\left[r +\\frac{1}{2} \\sigma^2 - \\lambda \\kappa, \\sigma, \\widetilde{\\lambda}, \\widetilde{p}, \\widetilde{\\eta}_1, \\widetilde{\\eta}_2; \\ln\\left(\\frac{K}{S_0}\\right), T \\right] - Ke^{-rT} \\Upsilon \\left[r -\\frac{1}{2}\\sigma^2 - \\lambda \\kappa, \\sigma, \\lambda, p,\\eta_1,\\eta_2; \\ln\\left(\\frac{K}{S_0} \\right), T \\right]

        where

        .. math::

            \\widetilde{p} = \\left(\\frac{p}{1+\\kappa}\\right) \\left(\\frac{\\eta_1}{\\eta_1 - 1}\\right), \\quad \\widetilde{\\eta}_1 = \\eta_1 - 1, \\quad \\widetilde{\\eta}_2 = \\eta_2 +1,

        .. math::

            \\widetilde{\\lambda}= \\lambda (\\kappa+1), \\quad \\kappa = \\frac{p\\eta_1}{\\eta_1 -1} + \\frac{q\\eta_2}{\\eta_2+1} - 1

        - For ``emm = "Esscher"``:

        Feature is not supported.

        References
        ----------
        .. [1] Kou, S. G. (2002). A jump-diffusion model for option pricing. Management science, 48(8), 1086-1101.
        """
        return self._closed_form_solution()
    
    def _closed_form_solution(self) -> Union[float, np.ndarray]:
        # implements the closed form calculations
        S0 = self.model.S0
        sigma = self.model.sigma
        lambda_ = self.model.lambda_
        eta1 = self.model.eta1
        eta2 = self.model.eta2
        p = self.model.p
        N = self.model.N
        r = self.option.r
        q = self.option.q
        K = self.option.K
        T = self.option.T
        payoff = self.option.payoff
        emm = self.option.emm

        if emm == "mean-correcting":

            def _P_function(n, k, p, eta1, eta2):
                if n == k:
                    series = p**n
                else:
                    series = 0
                    for i in range(k, n):
                        series = series + comb(n - k - 1, i - k) * comb(n, i) * (
                            eta1 / (eta1 + eta2)
                        ) ** (i - k) * (eta2 / (eta1 + eta2)) ** (n - i) * p**i * (
                            1 - p
                        ) ** (
                            n - i
                        )
                return series

            def _Q_function(n, k, p, eta1, eta2):
                if n == k:
                    series = (1 - p) ** n
                else:
                    series = 0
                    for i in range(k, n):
                        series = series + comb(n - k - 1, i - k) * comb(n, i) * (
                            eta1 / (eta1 + eta2)
                        ) ** (n - i) * (eta2 / (eta1 + eta2)) ** (i - k) * p ** (
                            n - i
                        ) * (
                            1 - p
                        ) ** (
                            i
                        )
                return series

            def _Hh(n, x):
                if n == -1:
                    result = np.sqrt(2 * np.pi) * st.norm.pdf(x)
                elif n == 0:
                    result = np.sqrt(2 * np.pi) * st.norm.cdf(-x)
                else:

                    def integrand(u):
                        return (u - x) ** n * np.exp(-(u**2) / 2)

                    result = (
                        1 / math.factorial(n) * integrate.quad(integrand, x, np.inf)[0]
                    )
                return result

            def _I_function(n, c, a, b, d):
                series = 0
                if b > 0 and a != 0:
                    for i in range(0, n + 1):
                        series = series + (b / a) ** (n - i) * _Hh(i, b * c - d)
                    result = -np.exp(a * c) / a * series + (b / a) ** (n + 1) * np.sqrt(
                        2 * np.pi
                    ) / b * np.exp(a * d / b + a**2 / (2 * b**2)) * st.norm.cdf(
                        -b * c + d + a / b
                    )

                elif b < 0 and a < 0:
                    for i in range(0, n + 1):
                        series = series + (b / a) ** (n - i) * _Hh(i, b * c - d)
                    result = -np.exp(a * c) / a * series - (b / a) ** (n + 1) * np.sqrt(
                        2 * np.pi
                    ) / b * np.exp(a * d / b + a**2 / (2 * b**2)) * st.norm.cdf(
                        b * c - d - a / b
                    )

                else:
                    result = 0
                return result

            def _Upsilon(u, ttm, mu, sigma, lamb, eta1, eta2, p, num_jumps):
                factor1 = np.exp((sigma * eta1) ** 2 * ttm / 2) / (
                    sigma * np.sqrt(2 * np.pi * ttm)
                )
                factor2 = np.exp((sigma * eta2) ** 2 * ttm / 2) / (
                    sigma * np.sqrt(2 * np.pi * ttm)
                )

                series1 = 0
                series2 = 0

                for n in range(1, num_jumps + 1):
                    poisson_pdf = (
                        np.exp(-lamb * ttm) * (lamb * ttm) ** n / math.factorial(n)
                    )
                    subseries1 = 0
                    subseries2 = 0
                    for k in range(1, n + 1):
                        subseries1 = subseries1 + (
                            sigma * eta1 * ttm**0.5
                        ) ** k * _P_function(n, k, p, eta1, eta2) * _I_function(
                            k - 1,
                            u - mu * ttm,
                            -eta1,
                            -1 / (sigma * ttm**0.5),
                            -sigma * eta1 * ttm**0.5,
                        )

                        subseries2 = subseries2 + (
                            sigma * eta2 * ttm**0.5
                        ) ** k * _Q_function(n, k, p, eta1, eta2) * _I_function(
                            k - 1,
                            u - mu * ttm,
                            eta2,
                            1 / (sigma * ttm**0.5),
                            -sigma * eta2 * ttm**0.5,
                        )
                    series1 = series1 + poisson_pdf * subseries1
                    series2 = series2 + poisson_pdf * subseries2

                result = (
                    factor1 * series1
                    + factor2 * series2
                    + np.exp(-lamb * ttm)
                    * st.norm.cdf(-(u - mu * ttm) / (sigma * ttm**0.5))
                )
                return result

            kappa = p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1

            p_tilde = (p / (1 + kappa)) * (eta1 / (eta1 - 1))

            lambda_tilde = lambda_ * (kappa + 1)

            eta1_tilde = eta1 - 1

            eta2_tilde = eta2 + 1

            x = np.log(K / (np.exp(-q*T)*S0))

            mu1 = r + sigma**2 / 2 - lambda_ * kappa
            mu2 = r - sigma**2 / 2 - lambda_ * kappa

            option_price = np.exp(-q*T)*S0 * _Upsilon(
                x,
                T,
                mu1,
                sigma,
                lambda_tilde,
                eta1_tilde,
                eta2_tilde,
                p_tilde,
                N,
            ) - K * np.exp(-r * T) * _Upsilon(
                x,
                T,
                mu2,
                sigma,
                lambda_,
                eta1,
                eta2,
                p,
                N,
            )
        elif emm == "Esscher":
            raise FeatureNotImplementedError("Esscher")
        else:
            raise UnsupportedEMMTypeError(emm)
        
        if payoff == 'c':
            return option_price
        elif payoff == 'p':
            return option_price + np.exp(-r*T) * K - np.exp(-q*T)*S0
        else:
            raise UnknownPayoffTypeError(payoff)