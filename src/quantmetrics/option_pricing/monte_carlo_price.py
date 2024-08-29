#option_pricing/monte_carlo_simulation.py

from quantmetrics.option_pricing import SimulatePaths

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option

class MonteCarloPrice:
    def __init__(
            self,
            model : 'LevyModel',
            option : 'Option'):
        """
        Initialize the CharacteristicFunction with a model and an option.

        Parameters:
        model (LevyModel): The Levy model.
        option (Option): The option parameters.
        """
        self.model = model
        self.option = option

    def calculate(self, num_timesteps : int=100, num_paths : int = 1000, seed :int = 1) -> float:
        """
        Calculate the characteristic function for the given model.

        Parameters:
        u (np.ndarray): Input array for the characteristic function.

        Returns:
        np.ndarray: The characteristic function values.
        """
        r = self.option.r
        K = self.option.K
        T = self.option.T
        path_object = SimulatePaths(self.model, self.option)


        paths = path_object.simulate(num_timesteps, num_paths, seed)

        S = paths["S"]
        payoff = np.maximum(S[-1, :] - K, 0)
        return np.mean(np.exp(-r * T ) * payoff)