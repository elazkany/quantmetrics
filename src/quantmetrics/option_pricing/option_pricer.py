#option_pricing/option_pricer.py
from quantmetrics.option_pricing import ExactSolution
from quantmetrics.option_pricing import FFTPrice
from quantmetrics.option_pricing import MonteCarloPrice

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option

class OptionPricer:
    def __init__(
            self,
            model : 'LevyModel',
            option : 'Option'):
        """
        Initialize the OptionPricer with a model and an option.

        Parameters:
        model (LevyModel): The Levy model.
        option (Option): The option parameters.
        """
        self.model = model
        self.option = option

    def exact(self) -> float:
        """
        Calculate the option price using the exact solution method.

        Returns:
        float: The calculated option price.
        """
        return ExactSolution(self.model, self.option).calculate()

    def fft(
            self,
            N : int = 2**12,
            eps : float = 1 / 150,
            alpha : float = 0.75
    )-> float:
        """
        Calculate the option price using the fast Fourier transform method.

        Parameters:
        N (int): Number of points for FFT.
        eps (float): Grid spacing for FFT.
        alpha (float): Damping factor for FFT.

        Returns:
        float: The calculated option price.
        """
        return FFTPrice(self.model, self.option).calculate(N=N, eps = eps, alpha = alpha)
    
    def monte_carlo(
            self,
            num_timesteps : int = 100,
            num_paths : int = 1000,
            seed : int = 1,
            ) -> float:
        return MonteCarloPrice(self.model, self.option).calculate(num_timesteps, num_paths, seed)
