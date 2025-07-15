#src\quantmetrics\price_calculators\ljd_pricing\ljd_closed_form.py

import numpy as np
import scipy.stats as st
import math
from typing import TYPE_CHECKING, Union

#from quantmetrics.risk_calculators.martingale_equation import RiskPremium
from quantmetrics.utils.exceptions import UnknownPayoffTypeError, FeatureNotImplementedError, UnsupportedEMMTypeError

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option

class LJDClosedForm:
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
        The closed-form solution for the European option depends on the equivalent martingale measure (EMM) considered.

        - For ``emm = "mean-correcting"``:
        
        This measure is the same in the Black-Scholes framework and the equation for the option price is the Merton jump-diffusion option price formula [3]_ where the jump risk is not priced and :math:`\\lambda^\\mathbb{Q}=\\lambda`. The formula for the European call option is:

        .. math::

            C =\\sum_{n\\geq 0} \\frac{e^{-\\lambda T}(\\lambda T)^n}{n!}\\left(e^{-qT} S^{(n)} \\ N(d_+^{(n)})  - e^{-rT}K \\ N(d_-^{(n)}) \\right)

        and for the European put option, we use the call-put parity:

        .. math::

            P = C + e^{-rT} K - e^{-qT}S_0

        where:
  
        
            .. math::
            
                d_+^{(n)} = \\frac{\\ln \\left(\\frac{e^{-qT}S^{(n)}}{K}\\right) + (r + \\frac{(\\sigma^{(n)})^2}{2})T}{\\sigma^{(n)} \\sqrt{T}},

            .. math::    

                d_-^{(n)} = d_+^{(n)} - \\sigma^{(n)} \\sqrt{T}, \quad \\kappa = \\exp\\left(\\mu_J + \\frac{\\sigma_J^2}{2} \\right) -1  

            .. math::

                S^{(n)} = S_0 \\exp \\left(n(\\mu_J + \\frac{\\sigma_J^2}{2}) - \\lambda \\kappa T \\right) \\quad \\text{and} \\quad  \\sigma^{(n)} = \\sqrt{\\sigma^2 + \\frac{n\\sigma_J^2}{T}}


            - :math:`C` is the call option price.
            - :math:`P` is the put option price.
            - :math:`q` is the dividend yield.
            - :math:`T` is the time to maturity.
            - :math:`S_0` is the underlying price.
            - :math:`r` is the risk-free interest rate.            
            - :math:`K` is the strike price.
            - :math:`\\sigma` is the volatility.
            - :math:`\\mu_J` is the mean of the jump sizes.
            - :math:`\\sigma_J` is the standard deviation of the jump sizes.
            - :math:`\\lambda` is the jump intensity rate.
            - :math:`N(x)` is the standard normal cumulative distribution function: :math:`N(x)= \\frac{1}{\\sqrt{2\pi}}\\int_{-\\infty}^x e^{-\\frac{u^2}{2}}du`.  

        - For ``emm = Esscher"``:
        
        Under the Esscher EMM, the jump risk is priced, see [2]_ for details.


        Examples
        --------
        >>> from quantmetrics.levy_models import LJD
        >>> from quantmetrics.option_pricing import Option, OptionPricer
        >>> ljd = LJD() # S0=50, sigma=0.2, lambda_=1, muJ=-0.1, sigmaJ=0.1
        >>> option = Option(K=np.array([20,50,80]), T = 20/252) # r=0.05, q=0.02
        >>> ljd_pricer = OptionPricer2(ljd, option)
        >>> ljd_pricer.closed_form()
        array([2.99999057e+01, 1.32940700e+00, 1.19634276e-07])

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

        if emm == "mean-correcting":
            lambda_Q = lambda_

            for n in range(0, N + 1):
                x_n = np.exp(-q*T) * S0 * np.exp(n * (muJ + sigmaJ**2 / 2) - lambda_Q * kappa * T)

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
        elif emm == "Esscher":
            raise FeatureNotImplementedError(emm)
        else:
            raise UnsupportedEMMTypeError(emm)

        if payoff == 'c':
            return option_price
        elif payoff == 'p':
            return option_price + np.exp(-r*T) * K - np.exp(-q*T)*S0
        else:
            raise UnknownPayoffTypeError(payoff)
