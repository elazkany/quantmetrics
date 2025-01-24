# option_pricing/exact_equation.py

from quantmetrics.levy_models import GBM, CJD, LJD, DEJD
from quantmetrics.option_pricing import RiskPremium

from typing import TYPE_CHECKING
import numpy as np
import scipy.stats as st
import math
from scipy.special import comb
import scipy.integrate as integrate

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
        elif isinstance(self.model, CJD):
            return self._cjd_exact_solution()
        elif isinstance(self.model, LJD):
            return self._ljd_exact_solution()
        elif isinstance(self.model, DEJD):
            return self._dejd_exact_solution()

    def _black_scholes_exact_price(self):
        """
        Calculate the European option price using the Black-Scholes exact equation.

        Returns
        -------
        float
            The calculated option price.

        References
        ----------
            Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of political economy, 81(3), 637-654.
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

    def _cjd_exact_solution(self):
        S0 = self.model.S0
        mu = self.model.mu
        sigma = self.model.sigma
        lambda_ = self.model.lambda_
        gamma = self.model.gamma
        N = self.model.N
        r = self.option.r
        q = self.option.q
        K = self.option.K
        T = self.option.T
        payoff = self.option.payoff
        emm = self.option.emm
        psi = self.option.psi

        gamma_tilde = np.exp(gamma) - 1

        if emm == "Black-Scholes":
            Lambda = lambda_
        else:
            theta = RiskPremium(self.model, self.option).calculate()
            Lambda = (r - mu - theta * sigma**2 + lambda_ * gamma_tilde) / gamma_tilde

        option_price = 0
        for n in range(0, N + 1):
            x_n = S0 * np.exp(n * gamma - Lambda * gamma_tilde * T)

            poisson_pdf = np.exp(-Lambda * T) * (Lambda * T) ** n / math.factorial(n)

            d_plus = (np.log(x_n / K) + (r + sigma**2 / 2) * T) / (sigma * T**0.5)

            d_minus = d_plus - sigma * T**0.5

            bs_option_price = x_n * st.norm.cdf(d_plus) - K * np.exp(
                -r * T
            ) * st.norm.cdf(d_minus)

            option_price = option_price + poisson_pdf * bs_option_price

        return option_price

    def _ljd_exact_solution(self):
        S0 = self.model.S0
        mu = self.model.mu
        sigma = self.model.sigma
        lambda_ = self.model.lambda_
        muJ = self.model.muJ
        sigmaJ = self.model.sigmaJ
        N = self.model.N
        r = self.option.r
        q = self.option.q
        K = self.option.K
        T = self.option.T
        payoff = self.option.payoff
        emm = self.option.emm
        psi = self.option.psi

        option_price = 0
        kappa = np.exp(muJ + sigmaJ**2 / 2) - 1

        if emm == "Black-Scholes":
            lambda_Q = lambda_

            for n in range(0, N + 1):
                x_n = S0 * np.exp(n * (muJ + sigmaJ**2 / 2) - lambda_Q * kappa * T)

                sigma_n = np.sqrt(sigma**2 + n * sigmaJ**2 / T)

                poisson_pdf = (
                    np.exp(-lambda_Q * T) * (lambda_Q * T) ** n / math.factorial(n)
                )

                d_plus = (np.log(x_n / K) + (r + sigma_n**2 / 2) * T) / (
                    sigma_n * T**0.5
                )

                d_minus = d_plus - sigma_n * T**0.5

                bs_option_price = x_n * st.norm.cdf(d_plus) - K * np.exp(
                    -r * T
                ) * st.norm.cdf(d_minus)

                option_price = option_price + poisson_pdf * bs_option_price
        else:
            pass  # TODO
        return option_price

    def _dejd_exact_solution(self):
        S0 = self.model.S0
        mu = self.model.mu
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
        psi = self.option.psi

        if emm == "Black-Scholes":

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

            zeta = p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1

            p_tilde = (p / (1 + zeta)) * (eta1 / (eta1 - 1))

            lambda_tilde = lambda_ * (zeta + 1)

            eta1_tilde = eta1 - 1

            eta2_tilde = eta2 + 1

            x = np.log(K / S0)

            mu1 = r + sigma**2 / 2 - lambda_ * zeta
            mu2 = r - sigma**2 / 2 - lambda_ * zeta

            option_price = S0 * _Upsilon(
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
        else:
            pass  # TODO
        return option_price
