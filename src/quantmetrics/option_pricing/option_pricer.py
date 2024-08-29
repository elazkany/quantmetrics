# option_pricing/option_pricer.py
from quantmetrics.option_pricing import ExactSolution
from quantmetrics.option_pricing import FFTPrice
from quantmetrics.option_pricing import MonteCarloPrice

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option


class OptionPricer:
    def __init__(self, model: "LevyModel", option: "Option"):
        """
        Initialize the OptionPricer with a model and an option.

        Parameters
        ----------
        model : LevyModel
            The Levy model used for pricing the option.
        option : Option
            The option parameters including interest rate, strike price, etc.
        """
        self.model = model
        self.option = option

    def exact(self) -> float:
        """
        Calculate the option price using the exact solution method.

        Returns
        -------
        float
            The calculated option price.
        """
        return ExactSolution(self.model, self.option).calculate()

    def fft(self, N: int = 2**12, eps: float = 1 / 150, alpha: float = 0.75) -> float:
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
        return FFTPrice(self.model, self.option).calculate(N=N, eps=eps, alpha=alpha)

    def monte_carlo(
        self,
        num_timesteps: int = 200,
        num_paths: int = 10000,
        seed: int = 42,
    ) -> float:
        """
        Calculate the option price using the Monte Carlo simulation method.

        Parameters
        ----------
        num_timesteps : int, optional
            Number of time steps for the simulation (default is 200).
        num_paths : int, optional
            Number of simulated paths (default is 10000).
        seed : int, optional
            Seed for random number generator (default is 42).

        Returns
        -------
        float
            The calculated option price.
        """
        return MonteCarloPrice(self.model, self.option).calculate(
            num_timesteps, num_paths, seed
        )
