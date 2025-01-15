# option_pricing/fft_price.py

from quantmetrics.option_pricing import CharacteristicFunction

from typing import TYPE_CHECKING
import numpy as np
from scipy.fft import fft

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option


class FFTPrice:
    def __init__(
        self,
        model: "LevyModel",
        option: "Option",
    ):
        """
        Initialize the FFTPrice with a model and an option.

        Parameters
        ----------
        model : LevyModel
            A Levy model used for pricing the option.
        option : Option
            The option parameters including interest rate, strike price, etc.
        """
        self.model = model
        self.option = option

    def calculate(
        self, N: int = 2**12, eps: float = 1 / 150, alpha: float = 0.75
    ) -> float:
        """
        Calculate the option price using the fast Fourier transform method.

        Parameters
        ----------
        N : int, optional
            Number of points for FFT (default is 2^12).
        eps : float, optional
            Grid spacing for FFT (default is 1/150).
        alpha : float, optional
            Damping factor for FFT (default is 0.75).

        Returns
        -------
        float
            The calculated option price.
        """
        S0 = self.model.S0
        r = self.option.r
        K = self.option.K
        T = self.option.T
        char_func = CharacteristicFunction(self.model, self.option)

        if np.isscalar(K):
            K = np.array([K])

        k = np.log(K / S0)
        x0 = np.log(S0 / S0)
        g = 2  # factor to increase accuracy
        N = g * 4096
        eps = (g * 150.0) ** -1
        eta = 2 * np.pi / (N * eps)
        b = 0.5 * N * eps - k
        u = np.arange(1, N + 1, 1)
        vo = eta * (u - 1)

        prices = np.array([])

        for i in range(0, len(K)):
            # Modifications to ensure integrability
            if S0 >= 0.95 * K[i]:  # ITM case
                alpha = 1.5
                omega = vo - (alpha + 1) * 1j
                modcharFunc = np.exp(-r * T) * (
                    char_func.calculate(omega)
                    / (alpha**2 + alpha - vo**2 + 1j * (2 * alpha + 1) * vo)
                )
            else:  # OTM case
                alpha = 1.1
                omega = (vo - 1j * alpha) - 1j
                modcharFunc1 = np.exp(-r * T) * (
                    1 / (1 + 1j * (vo - 1j * alpha))
                    - np.exp(r * T) / (1j * (vo - 1j * alpha))
                    - char_func.calculate(omega)
                    / ((vo - 1j * alpha) ** 2 - 1j * (vo - 1j * alpha))
                )
                omega = (vo + 1j * alpha) - 1j
                modcharFunc2 = np.exp(-r * T) * (
                    1 / (1 + 1j * (vo + 1j * alpha))
                    - np.exp(r * T) / (1j * (vo + 1j * alpha))
                    - char_func.calculate(omega)
                    / ((vo + 1j * alpha) ** 2 - 1j * (vo + 1j * alpha))
                )
            # Numerical FFT Routine
            delt = np.zeros(N)  # , dtype=np.float)
            delt[0] = 1
            j = np.arange(1, N + 1, 1)
            SimpsonW = (3 + (-1) ** j - delt) / 3
            if S0 >= 0.95 * K[i]:
                FFTFunc = np.exp(1j * b[i] * vo) * modcharFunc * eta * SimpsonW
                payoff = (fft(FFTFunc)).real
                CallValueM = np.exp(-alpha * k[i]) / np.pi * payoff
            else:
                FFTFunc = (
                    np.exp(1j * b[i] * vo)
                    * (modcharFunc1 - modcharFunc2)
                    * 0.5
                    * eta
                    * SimpsonW
                )
                payoff = (fft(FFTFunc)).real
                CallValueM = payoff / (np.sinh(alpha * k[i]) * np.pi)
            pos = int((k[i] + b[i]) / eps)
            CallValue = CallValueM[pos] * S0
            # klist = np.exp((np.arange(0, N, 1) - 1) * eps - b) * S0
            if CallValue <= 0.0:
                prices = np.append(prices, 0.0)
            else:
                prices = np.append(prices, CallValue)  # , klist[pos - 50:pos + 50]

        return prices
