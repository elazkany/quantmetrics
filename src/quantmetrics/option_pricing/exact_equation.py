# option_pricing/exact_equation.py

from quantmetrics.levy_models import GBM, CJD, LJD
from quantmetrics.option_pricing import RiskPremium

from typing import TYPE_CHECKING
import numpy as np
import scipy.stats as st
import math

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
            Lambda = (r - mu - theta *sigma**2 +lambda_*gamma_tilde) / gamma_tilde

        option_price = 0
        for n in range(0, N + 1):
            x_n = S0 * np.exp(
                n * gamma - Lambda * gamma_tilde * T
                )

            poisson_pdf = (
                np.exp(-Lambda *  T)
                * (Lambda * T) ** n
                / math.factorial(n)
                )

            d_plus = (
                np.log(x_n / K)
                + (r + sigma**2 / 2) * T
                ) / (sigma * T**0.5)

            d_minus = d_plus - sigma * T**0.5

            bs_option_price = x_n * st.norm.cdf(
                    d_plus
                ) - K * np.exp(-r * T) * st.norm.cdf(
                    d_minus
                )

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

        if emm == "Black-Scholes":
            Lambda = lambda_
        else:
            # TODO:
            pass

        option_price = 0
        for n in range(0, N + 1):
            x_n = S0 * np.exp(
                n * (muJ + sigmaJ**2 / 2)
                - Lambda
                * (np.exp(muJ + sigmaJ**2 / 2) - 1)
                * T
            )

            sigma_n = np.sqrt(
                sigma**2 + n * sigmaJ**2 / T
                )

            poisson_pdf = (
                np.exp(-Lambda * T)
                * (Lambda * T) ** n
                / math.factorial(n)
                )

            d_plus = (
                np.log(x_n / K)
                + (r + sigma_n**2 / 2) * T
                ) / (sigma_n * T**0.5)

            d_minus = d_plus - sigma_n * T**0.5

            bs_option_price = x_n * st.norm.cdf(
                    d_plus
                ) - K * np.exp(
                    -r * T
                ) * st.norm.cdf(
                    d_minus
                )

            option_price = option_price + poisson_pdf * bs_option_price

        return option_price
