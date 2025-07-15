import numpy as np
from scipy.stats import norm
from scipy.special import gamma

from scipy.integrate import quad

from typing import TYPE_CHECKING, Union
from quantmetrics.utils.exceptions import UnknownPayoffTypeError

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option


class VGClosedForm:
    """
        Implements the closed-form "numerical" solution for pricing European options under a Variance Gamma (VG) model.

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
        Calculate the European option price.

        Returns
        -------
        Union[float, np.ndarray]
            The calculated option price.

        Notes
        -------
        The closed-form solution for the European call option:

        .. math::

            C = e^{-qT}S_0 \\ \\Psi\\left(d\\sqrt{\\frac{1-c_1}{\\kappa}}, (\\alpha+s)\\sqrt{\\frac{\\kappa}{1-c_1}},\\frac{T}{\\kappa}\\right) - e^{-rT}K \\ \\Psi\\left(d\\sqrt{\\frac{1-c_2}{\\kappa}}, \\alpha \\ \\sqrt{\\frac{\\kappa}{1-c_2}}, \\frac{T}{\\kappa}\\right)

        and for the European put option the call-put parity:

        .. math::

            

        Where:

            .. math::
            
                \\Psi(a,b,\\gamma) = \\int_0^\\infty N \\left(\\frac{a}{\\sqrt{u}} + b \\sqrt{u} \\right) \\frac{u^{\\gamma -1} e^{-u}}{\\Gamma(\\gamma)}du,

            .. math::

                d = \\frac{1}{s} \\left[\\ln\\left(\\frac{S_0}{K} + (r-q)T + \\frac{T}{\\kappa} \\ln \\left(\\frac{1-c_1}{1-c_2} \\right) \\right) \\right], \\quad \\alpha = \\frac{m}{\\sqrt{\\delta^2 + \\frac{m^2 \\kappa}{2}}}, \\quad s = \\frac{\\delta}{1+\\frac{m^2 \\kappa}{2\\delta^2}}, \\quad c_1 = \\frac{\\kappa}{2}(\\alpha + s)^2, \\quad c_2 = \\frac{\\kappa}{2} \\alpha^2
                    
            - :math:`C` is the call option price.
            - :math:`P` is the put option price.
            - :math:`q` is the dividend yield.
            - :math:`T` is the time to maturity.
            - :math:`S_0` is the underlying price.
            - :math:`r` is the risk-free interest rate.            
            - :math:`K` is the strike price.
            - :math:`\\delta` is the volatility of the subordinated Brownian motion.
            - :math:`m` is the drift of the subordinated Brownian motion.
            - :math:`\\kappa` is the variance rate of the subordinator Gamma process.
            - :math:`N(x)` is the standard normal cumulative distribution function: :math:`N(x)= \\frac{1}{\\sqrt{2\pi}}\\int_{-\\infty}^x e^{-\\frac{u^2}{2}}du`.

        """
        return self._closed_form_solution()
    
    def _closed_form_solution(self) -> Union[float, np.ndarray]:
        
        S0 = self.model.S0
        m = self.model.m
        delta = self.model.delta
        kappa = self.model.kappa
        r = self.option.r
        q = self.option.q
        K = self.option.K
        T = self.option.T
        payoff = self.option.payoff

        # Compute α and s
        alpha = -m / np.sqrt(delta**2 + (m**2 * kappa) / 2)
        s = delta / np.sqrt(1 + (m**2 * kappa) / (2 * delta**2))

        # Compute c1 and c2
        c1 = (kappa / 2) * (alpha + s)**2
        c2 = (kappa / 2) * alpha**2

        # Compute d
        d = (1 / s) * (
            np.log(S0 / K)
            + (r - q) * T
            + (T / kappa) * np.log((1 - c1) / (1 - c2))
        )

        # Compute Ψ terms
        psi_1 = self._Psi_function(
            a=d * np.sqrt((1 - c1) / kappa),
            b=(alpha + s) * np.sqrt(kappa / (1 - c1)),
            gamma_param=T / kappa
        )

        psi_2 = self._Psi_function(
            a=d * np.sqrt((1 - c2) / kappa),
            b=alpha *  np.sqrt(kappa / (1 - c2)),
            gamma_param=T / kappa
        )

        call_price = (
            np.exp(-q * T) * S0 * psi_1
            - np.exp(-r * T) * K * psi_2
        )

        # Calculate the option price based on the payoff type
        if payoff == "c":
            option_price = call_price
        elif payoff == "p":
            option_price = call_price + np.exp(-r*T) * K - np.exp(-q*T)*S0
        else:
            raise UnknownPayoffTypeError(payoff)
        return option_price
    
    def _Psi_function(self, a, b, gamma_param):
        """
        Compute the Ψ(a, b, γ) function as defined in the Variance Gamma model.

        Parameters:
        -----------
        a : float
            Parameter related to d and c_1 or c_2.
        b : float
            Parameter related to α and s.
        gamma_param : float
            Time-related parameter, T / κ.

        Returns:
        --------"
        float
            The integral Ψ(a, b, γ).
        """
        integrand = lambda u: norm.cdf(a / np.sqrt(u) + b * np.sqrt(u)) * u**(gamma_param-1)* np.exp(-u) / gamma(gamma_param)
        
        y, err = quad(integrand, 0, np.inf)

        return y

