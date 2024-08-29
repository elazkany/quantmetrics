#option_pricing/path_simulator.py

from quantmetrics.levy_models import GBM

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option

class SimulatePaths:
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

    def simulate(self,
                num_timesteps: int = 200,
                num_paths: int = 1000,
                seed: int = 1) -> dict:
        """
        Calculate the characteristic function for the given model.

        Parameters:
        u (np.ndarray): Input array for the characteristic function.

        Returns:
        np.ndarray: The characteristic function values.
        """
        if isinstance(self.model, GBM):
            return self._gbm_paths(num_timesteps, num_paths, seed)

    def _gbm_paths(self, num_timesteps, num_paths, seed):
        """
        Calculate the characteristic function for the GBM model.

        Parameters:
        u (np.ndarray): Input array for the characteristic function.

        Returns:
        np.ndarray: The characteristic function values.
        """
        np.random.seed(seed)

        S0 = self.model.S0
        sigma = self.model.sigma
        r = self.option.r
        q = self.option.q
        T = self.option.T

        dt = T / float(num_timesteps)
        Z = np.random.standard_normal(
            size=(num_timesteps, num_paths)
        )  # generate standard normal random variable of size.... [t,omega]
        # Standard Brownian motion
        W = np.zeros((num_timesteps + 1, num_paths))
        # Return process under Q (risk-neutral)
        X = np.zeros(W.shape)
        X[0, :] = np.log(S0)
        time = np.zeros(W.shape[0])

        for i in range(0, num_timesteps):
            # Making sure that samples from the normal distribution have mean 0 and variance 1
            if num_paths > 1:
                Z[i, :] = (Z[i, :] - np.mean(Z[i, :])) / np.std(Z[i, :])
            W[i + 1, :] = W[i, :] + np.power(dt, 0.5) * Z[i, :]
            X[i + 1, :] = (
                X[i, :]
                + (r - q - 0.5 * sigma**2) * dt
                + sigma * (W[i + 1, :] - W[i, :])
            )
            time[i + 1] = time[i] + dt

        # Compute exponent of ABM
        S = np.exp(X)
        paths = {"time": time, "S": S}
        return paths
