#src\quantmetrics\price_calculators\cjd_pricing\cjd_closed_form.py

import numpy as np
import scipy.stats as st
import math
from typing import TYPE_CHECKING, Union

from quantmetrics.risk_calculators.martingale_equation import RiskPremium

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option

class CJDClosedForm:
    """
        Implements the closed-form solution for pricing European options under a constant jump-diffusion (CJD) model.

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
        Calculate the European option price using the constant jump-diffusion exact equation.

        Returns
        -------
        Union[float, np.ndarray]
            The calculated option price.
        
        Notes
        -------
        This method computes the closed-form solution for European call options under a jump-diffusion model. The specific formula depends on the equivalent martingale measure (EMM) chosen:

        .. math::

            C =\\sum_{n\\geq 0} \\frac{e^{-\\lambda^\\mathbb{Q} T}(\\lambda^\\mathbb{Q} T)^n}{n!}\\left(e^{-qT} S^{(n)} \\ N(d_+^{(n)})  - e^{-rT}K \\ N(d_-^{(n)}) \\right)

        - For ``emm = "mean-correcting"``:

        .. math::

            \\lambda^\\mathbb{Q} = \\lambda

        - For ``emm = "Esscher"``:

        .. math::

            \\lambda^\\mathbb{Q} = \\lambda \\exp \\left(\\theta \\gamma + \\psi \\gamma^2 \\right)

        where the first-order Esscher parameter :math:`\\theta` is the risk premium (market price of risk) and which is the unique solution to the martingale equation for each :math:`\\psi` which is the second-order Esscher parameter. See the documentation of the ``RiskPremium`` class for the martingale equation and refer to [1]_ for more details.
        
        For the European put option, we use the call-put parity:

        .. math::

            P = C + e^{-rT} K - e^{-qT}S_0

        In these formulas:  
        
        - .. math:: d_+^{(n)} = \\frac{\\ln \\left(\\frac{e^{-qT}S^{(n)}}{K}\\right) + (r + \\frac{(\\sigma^2}{2})T}{\\sigma \\sqrt{T}},

        - .. math:: d_-^{(n)} = d_+^{(n)} - \\sigma \\sqrt{T}, \quad \\kappa = e^\\gamma -1  

        - .. math:: S^{(n)} = S_0 \\exp \\left(n\\gamma - \\lambda^\\mathbb{Q} \\kappa T \\right) 

        The parameters are defined as follows:

        - :math:`C` is the call option price.
        - :math:`P` is the put option price.
        - :math:`q` is the dividend yield.
        - :math:`T` is the time to maturity.
        - :math:`S_0` is the underlying price.
        - :math:`r` is the risk-free interest rate.            
        - :math:`K` is the strike price.
        - :math:`\\sigma` is the volatility.
        - :math:`\\gamma` is the jump size.
        - :math:`\\lambda` is the jump intensity rate.
        - :math:`N(x)` is the standard normal cumulative distribution function: :math:`N(x)= \\frac{1}{\\sqrt{2\pi}}\\int_{-\\infty}^x e^{-\\frac{u^2}{2}}du`.  
        
        Examples
        --------
        >>> from quantmetrics.levy_models import CJD
        >>> from quantmetrics.option_pricing import Option, OptionPricer
        >>> cjd = CJD() # S0=50, sigma=0.2, lambda_=1, gamma=-0.1
        >>> option = Option(K=np.array([20,50,80]), T = 20/252) # r=0.05, q=0.02
        >>> cjd_pricer = OptionPricer(cjd, option)
        >>> cjd_pricer.closed_form()
        array([2.99999057e+01, 1.28936407e+00, 6.27880054e-17])

        References
        ----------

        .. [1] Choulli, T., Elazkany, E., & Vanmaele, M. (2024). Applications of the Second-Order Esscher Pricing in Risk Management. arXiv preprint arXiv:2410.21649.
        
        .. [2] Matsuda, K. (2004). Introduction to option pricing with Fourier transform: Option pricing with exponential Lévy models. Department of Economics The Graduate Center, The City University of New York, 1-241.

        .. [3] Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. Journal of financial economics, 3(1-2), 125-144.

        """
        return self._closed_form_solution()
    
    def _closed_form_solution(self) -> Union[float, np.ndarray]:
        # implements the closed form calculations
        S0 = self.model.S0
        mu = self.model._mu
        sigma = self.model._sigma
        lambda_ = self.model._lambda_
        gamma = self.model._gamma
        N = self.model.N
        r = self.option.r
        q = self.option.q
        K = self.option.K
        T = self.option.T
        payoff = self.option.payoff
        emm = self.option.emm

        gamma_tilde = np.exp(gamma) - 1

        if emm == "mean-correcting":
            Lambda = lambda_
        elif emm == "Esscher":
            theta = RiskPremium(self.model, self.option).calculate()
            Lambda = (r - mu - theta * sigma**2 + lambda_ * gamma_tilde) / gamma_tilde
        else:
            raise ValueError(f"Unknown or unsupported EMM type: {emm}. EMM is either 'mean-correcting' or 'Esscher'.")

        option_price = 0
        for n in range(0, N + 1):
            x_n = np.exp(-q*T)*S0 * np.exp(n * gamma - Lambda * gamma_tilde * T)

            poisson_pdf = np.exp(-Lambda * T) * (Lambda * T) ** n / math.factorial(n)

            d_plus = (np.log(x_n / K) + (r + sigma**2 / 2) * T) / (sigma * T**0.5)

            d_minus = d_plus - sigma * T**0.5

            bs_option_price = x_n * st.norm.cdf(d_plus) - K * np.exp(
                -r * T
            ) * st.norm.cdf(d_minus)

            option_price = option_price + poisson_pdf * bs_option_price

        if payoff == 'c':
            return option_price
        elif payoff == 'p':
            return option_price + np.exp(-r*T) * K - np.exp(-q*T)*S0
        else:
            raise ValueError(f"Unknown payoff type: {payoff}. payoff is either 'c' for call or 'p' for put.")