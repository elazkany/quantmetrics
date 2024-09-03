# option_pricing/path_simulator.py

from quantmetrics.levy_models import GBM, CJD, LJD
from quantmetrics.option_pricing import RiskPremium

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
        elif isinstance(self.model, CJD):
            return self._cjd_paths(num_timesteps, num_paths, seed)
        elif isinstance(self.model, LJD):
            return self._ljd_paths(num_timesteps, num_paths, seed)
        else:
            pass

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

        # Euler approximation of the stock SDE
        S_Euler = np.zeros([num_timesteps+1, num_paths])
        S_Euler[0, :] = S0

        for i in range(0, num_timesteps):
            # Making sure that samples from the normal distribution have mean 0 and variance 1
            if num_paths > 1:
                Z[i, :] = (Z[i, :] - np.mean(Z[i, :])) / np.std(Z[i, :])
            W[i + 1, :] = W[i, :] + np.power(dt, 0.5) * Z[i, :]
            # Exact formula of the stock equation has a return process
            X[i + 1, :] = (
                X[i, :]
                + (r - q - 0.5 * sigma**2) * dt
                + sigma * (W[i + 1, :] - W[i, :])
            )

            # Euler approximation
            S_Euler[i+1,:] = S_Euler[i,:] + r * S_Euler[i,:]*dt + sigma * S_Euler[i,:] * (W[i + 1, :] - W[i, :])

            time[i + 1] = time[i] + dt

        # Compute exponent of ABM
        S = np.exp(X)
        paths = {"time": time, "S": S, "S_Euler":S_Euler}
        return paths
    
    def _cjd_paths(self, num_timesteps, num_paths, seed):
        S0 = self.model.S0
        mu = self.model.mu
        sigma = self.model.sigma
        lambda_ = self.model.lambda_
        gamma = self.model.gamma
        r = self.option.r
        T = self.option.T
        emm = self.option.emm
        psi = self.option.psi

        np.random.seed(seed)

        dt = T / float(num_timesteps)

        W = np.zeros((num_timesteps + 1, num_paths))
        X = np.zeros([num_timesteps + 1, num_paths])
        S = np.zeros([num_timesteps + 1, num_paths])
        time = np.zeros([num_timesteps + 1])

        
        X[0, :] = np.log(S0)
        S[0, :] = S0

        gamma_tilde = np.exp(gamma) - 1

        if emm == "Black-Scholes":
            Lambda = lambda_
        else:
            theta = RiskPremium(self.model, self.option).calculate()
            Lambda = (r - mu - theta *sigma**2 +lambda_*gamma_tilde) / gamma_tilde

        # Check this
        ZPois = np.random.poisson(Lambda * dt, [num_timesteps, num_paths])

        Z = np.random.normal(0.0, 1.0, [num_timesteps, num_paths])


        for i in range(0, num_timesteps):
                # Making sure that samples from a normal have mean 0 and variance 1
            if num_paths > 1:
                Z[i, :] = (Z[i, :] - np.mean(Z[i, :])) / np.std(Z[i, :])
                # Making sure that samples from a normal have mean 0 and variance 1
            W[i + 1, :] = W[i, :] + np.power(dt, 0.5) * Z[i, :]
            X[i + 1, :] = (
                (
                    X[i, :]
                    + (r - Lambda* gamma_tilde - 0.5 * sigma**2) * dt
                )
                + sigma * (W[i + 1, :] - W[i, :])
                + gamma * ZPois[i, :]
            )

            time[i + 1] = time[i] + dt

        S = np.exp(X)
        paths = {"time": time, "X": X, "S": S}
        return paths
    
    def _ljd_paths(self, num_timesteps, num_paths, seed):
        S0 = self.model.S0
        sigma = self.model.sigma
        lambda_ = self.model.lambda_
        muJ = self.model.muJ
        sigmaJ = self.model.sigmaJ
        r = self.option.r
        T = self.option.T

        np.random.seed(seed)

        dt = T / float(num_timesteps)

        W = np.zeros((num_timesteps + 1, num_paths))
        X = np.zeros([num_timesteps + 1, num_paths])
        S = np.zeros([num_timesteps + 1, num_paths])
        time = np.zeros([num_timesteps + 1])

        
        X[0, :] = np.log(S0)
        S[0, :] = S0

        kappa = np.exp(muJ + 0.5 * sigmaJ**2) - 1
        
        ZPois = np.random.poisson(lambda_ * dt, [num_timesteps, num_paths])

        Z = np.random.normal(0.0, 1.0, [num_timesteps, num_paths])

        J = np.random.normal(muJ, sigmaJ, [num_timesteps, num_paths])

        for i in range(0, num_timesteps):
                # Making sure that samples from a normal have mean 0 and variance 1
            if num_paths > 1:
                Z[i, :] = (Z[i, :] - np.mean(Z[i, :])) / np.std(Z[i, :])
                # Making sure that samples from a normal have mean 0 and variance 1
            W[i + 1, :] = W[i, :] + np.power(dt, 0.5) * Z[i, :]
            X[i + 1, :] = (
                (
                    X[i, :]
                    + (r - lambda_ * kappa - 0.5 * sigma**2) * dt
                )
                + sigma * (W[i + 1, :] - W[i, :])
                + J[i, :] * ZPois[i, :]
            )

            time[i + 1] = time[i] + dt

        S = np.exp(X)
        paths = {"time": time, "X": X, "S": S}
        return paths

