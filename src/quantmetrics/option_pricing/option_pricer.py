import numpy as np
from scipy.fft import fft

from quantmetrics.levy_models import GBM, CJD, LJD, DEJD, VG
from quantmetrics.price_calculators.gbm_pricing.gbm_calculator import GBMCalculator
from quantmetrics.price_calculators.cjd_pricing.cjd_calculator import CJDCalculator
from quantmetrics.price_calculators.ljd_pricing.ljd_calculator import LJDCalculator
from quantmetrics.price_calculators.dejd_pricing.dejd_calculator import DEJDCalculator
from quantmetrics.price_calculators.vg_pricing.vg_calculator import VGCalculator

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option
    from quantmetrics.price_calculators import BaseCalculator

class OptionPricer:
    def __init__(self, model: "LevyModel", option: "Option"):
        self.model = model
        self.option = option
        self.calculator = self._get_calculator()

    def _get_calculator(self) -> "BaseCalculator":
        if isinstance(self.model, GBM):
            return GBMCalculator(self.model, self.option)
        
        elif isinstance(self.model, CJD):
            return CJDCalculator(self.model, self.option)
        
        elif isinstance(self.model, LJD):
            return LJDCalculator(self.model, self.option)
        
        elif isinstance(self.model, DEJD):
            return DEJDCalculator(self.model, self.option)
        
        elif isinstance(self.model, VG):
            return VGCalculator(self.model, self.option)

        else:
            raise ValueError(f"Unknown model type: {type(self.model).__name__}")
        
    """
        
        
        
        elif isinstance(self.model, NIG):
            return NIGCalculator(self.model, self.option)
        elif isinstance(self.model, HestonModel):
            return HestonCalculator(self.model, self.option)
        """
        

    def closed_form(self) -> Union[float, np.ndarray]:
        return self.calculator.calculate_closed_form()

    def fft(self, N: int = 2**12, eps: float = 1/150, alpha_itm: float = 1.5, alpha_otm : float = 1.1) -> Union[float, np.ndarray]:
        """
        Calculate the option price using the fast Fourier transform method.

        Parameters
        ----------
        N : int, optional
            Number of points for FFT (default is 2^12).
        eps : float, optional
            Grid spacing for FFT (default is 1/150).
        alpha_itm : float, optional
            Damping factor for FFT (default is 0.75).

        Returns
        -------
        Union[float, np.ndarray]
            The calculated option prices.
        """
        S0 = self.model.S0
        r = self.option.r
        K = self.option.K
        T = self.option.T
        q = self.option.q

        if np.isscalar(K):
            K = np.array([K])

        k = np.log(K / S0 * np.exp(q*T))
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
            if S0 * np.exp(-q*T) >= 0.95 * K[i]:  # ITM case
                omega = vo - (alpha_itm + 1) * 1j
                modcharFunc = np.exp(-r * T) * (
                    self.calculator.calculate_characteristic_function(omega)
                    / (alpha_itm**2 + alpha_itm - vo**2 + 1j * (2 * alpha_itm + 1) * vo)
                )
            else:  # OTM case
                omega = (vo - 1j * alpha_otm) - 1j
                modcharFunc1 = np.exp(-r * T) * (
                    1 / (1 + 1j * (vo - 1j * alpha_otm))
                    - np.exp(r * T) / (1j * (vo - 1j * alpha_otm))
                    - self.calculator.calculate_characteristic_function(omega)
                    / ((vo - 1j * alpha_otm) ** 2 - 1j * (vo - 1j * alpha_otm))
                )
                omega = (vo + 1j * alpha_otm) - 1j
                modcharFunc2 = np.exp(-r * T) * (
                    1 / (1 + 1j * (vo + 1j * alpha_otm))
                    - np.exp(r * T) / (1j * (vo + 1j * alpha_otm))
                    - self.calculator.calculate_characteristic_function(omega)
                    / ((vo + 1j * alpha_otm) ** 2 - 1j * (vo + 1j * alpha_otm))
                )
            # Numerical FFT Routine
            delt = np.zeros(N)  # , dtype=np.float)
            delt[0] = 1
            j = np.arange(1, N + 1, 1)
            SimpsonW = (3 + (-1) ** j - delt) / 3
            if S0 * np.exp(-q*T) >= 0.95 * K[i]:
                FFTFunc = np.exp(1j * b[i] * vo) * modcharFunc * eta * SimpsonW
                payoff = (fft(FFTFunc)).real
                CallValueM = np.exp(-alpha_itm * k[i]) / np.pi * payoff
            else:
                FFTFunc = (
                    np.exp(1j * b[i] * vo)
                    * (modcharFunc1 - modcharFunc2)
                    * 0.5
                    * eta
                    * SimpsonW
                )
                payoff = (fft(FFTFunc)).real
                CallValueM = payoff / (np.sinh(alpha_otm * k[i]) * np.pi)
            pos = int((k[i] + b[i]) / eps)
            CallValue = CallValueM[pos] * S0 * np.exp(-q*T)
            # klist = np.exp((np.arange(0, N, 1) - 1) * eps - b) * S0
            if CallValue <= 0.0:
                prices = np.append(prices, 0.0)
            else:
                prices = np.append(prices, CallValue)  # , klist[pos - 50:pos + 50]

        return prices
        
    def monte_carlo(self, num_timesteps: int = 200, num_paths: int = 10000, seed: int = 42, exact_solution = True) -> Union[float, np.ndarray]:
        """
        Calculate the Monte Carlo price for the given option.

        Parameters
        ----------
        num_timesteps : int, optional
            Number of time steps (default is 200).
        num_paths : int, optional
            Number of simulated paths (default is 10000).
        seed : int, optional
            Seed for random number generator (default is 42).

        print_time : bool, optional
        Whether to print the elapsed time (default is False).

        Returns
        -------
        np.ndarray
            A two-dimensional array containing the estimated price of the option and the standard error.
        """
        r = self.option.r
        K = self.option.K
        T = self.option.T

        paths = self.calculator.simulate_paths_Q(num_timesteps, num_paths, seed)

        if exact_solution == True:
            S = paths["S_exact"]
        else:
            S = paths["S_euler"]
        
        payoff = np.maximum(S[:,-1].reshape(-1,1) - K, 0)

        mc_price = np.mean(np.exp(-r * T) * payoff, axis=0)

        # Calculate the sample variance
        sample_var = np.sum((payoff - mc_price) ** 2, axis=0) / (num_paths - 1)

        # Calculate the standard error
        standard_error = (sample_var / num_paths) ** 0.5

        return np.array([mc_price, standard_error])
