# option_pricing/path_simulator.py

from quantmetrics.levy_models import GBM

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option


class SimulatePaths:
    def __init__(self, model: "LevyModel", option: "Option"):
        """
        Initialize the SimulatePaths with a model and an option.

        Parameters
        ----------
        model : LevyModel
            The Levy model used for path simulation.
        option : Option
            The option parameters including interest rate, dividend yield, etc.
        """
        self.model = model
        self.option = option

    def simulate(
        self, num_timesteps: int = 200, num_paths: int = 10000, seed: int = 42
    ) -> dict:
        """
        Simulates paths for the given model.

        Parameters
        ----------
        num_timesteps : int, optional
            Number of time steps (default is 200).
        num_paths : int, optional
            Number of simulated paths (default is 10000).
        seed : int, optional
            Seed for random number generator (default is 42).

        Returns
        -------
        dict
            Dictionary containing the time steps and simulated paths.
        """
        if isinstance(self.model, GBM):
            return self._gbm_paths(num_timesteps, num_paths, seed)

    def _gbm_paths(self, num_timesteps, num_paths, seed):
        """
        Generate paths for the Geometric Brownian Motion (GBM) model.

        Parameters
        ----------
        num_timesteps : int
            Number of time steps.
        num_paths : int
            Number of simulated paths.
        seed : int
            Seed for random number generator.

        Returns
        -------
        dict
            Dictionary containing the time steps and simulated paths.

        References
        ---------
            Oosterlee, C. W., & Grzelak, L. A. (2019). Mathematical modeling and computation in finance: With exercises and python and matlab computer codes (1st ed.). World Scientific Publishing Co. Pte. Ltd.
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
