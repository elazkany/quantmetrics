# option_pricing/monte_carlo_simulation.py

from quantmetrics.option_pricing import SimulatePaths

from typing import TYPE_CHECKING
import numpy as np
import time

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option


class MonteCarloPrice:
    def __init__(self, model: "LevyModel", option: "Option"):
        """
        Initialize the MonteCarloPrice with a model and an option.

        Parameters
        ----------
        model : LevyModel
            The Levy model used for path simulation.
        option : Option
            The option parameters including interest rate, strike price, etc.
        """
        self.model = model
        self.option = option

    def calculate(
        self,
        num_timesteps: int = 200,
        num_paths: int = 10000,
        seed: int = 42,
        sde="exact",
        print_time: bool = False,
    ) -> np.ndarray:
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

        start_time = time.time()

        path_object = SimulatePaths(self.model, self.option)

        paths = path_object.simulate(num_timesteps, num_paths, seed)

        if sde == "exact":
            S = paths["S"]
        elif sde == "euler":
            S = paths["S_Euler"]
        else:
            pass

        payoff = np.maximum(S[-1, :] - K, 0)

        mc_price = np.mean(np.exp(-r * T) * payoff)

        end_time = time.time()

        elapsed_time = end_time - start_time

        # Calculate the sample variance
        sample_var = np.sum((payoff - mc_price) ** 2) / (num_paths - 1)

        # Calculate the standard error
        standard_error = (sample_var / num_paths) ** 0.5

        if print_time:
            print(
                f"Elapsed time : {elapsed_time} seconds   |   Standard error = {standard_error}"
            )

        return np.array([mc_price, standard_error])
