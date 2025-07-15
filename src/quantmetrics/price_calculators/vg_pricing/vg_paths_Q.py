from quantmetrics.risk_calculators.martingale_equation import RiskPremium
import numpy as np
from typing import TYPE_CHECKING

from quantmetrics.utils.exceptions import UnsupportedEMMTypeError, FeatureNotImplementedError

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option

class VGSimulatePathsQ:
    def __init__(self, model: "LevyModel", option: "Option"):
        """
        Initialize the VGSimulatePathsQ with a model and an option.

        Parameters
        ----------
        model : LevyModel
            The Levy model used for calculating the characteristic function.
        option : Option
            The option parameters including interest rate, volatility, etc.
        """
        self.model = model
        self.option = option

    def simulate(self, num_timesteps: int, num_paths: int, seed: int) -> np.ndarray:
        """
        
        Returns
        -------
        np.ndarray
            The characteristic function values.
        """
        return self._vg_simulate_paths_Q(num_timesteps, num_paths, seed)

    def _vg_simulate_paths_Q(self, num_timesteps: int, num_paths: int, seed: int) -> np.ndarray:
        """
        Generate paths for the lognormal jump-diffusion (LJD) model.

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

        """
        np.random.seed(seed)

        S0 = self.model.S0
        m = self.model.m
        delta = self.model.delta
        kappa = self.model.kappa
        r = self.option.r
        q = self.option.q
        T = self.option.T
        emm = self.option.emm
        psi = self.option.psi

        np.random.seed(seed)

        dt = T / float(num_timesteps)
        
        # Generate normally distributed random variables with shape (num_paths, num_timesteps)
        Z = np.random.standard_normal(size=(num_paths, num_timesteps))
        
        # Normalize each column so that each time-step's ensemble has mean 0 and variance 1.
        Z = (Z - np.mean(Z, axis=0)) / np.std(Z, axis=0)

        # Generate Gamma distributed random variables
        dG = np.random.gamma(dt/kappa, kappa, size=(num_paths, num_timesteps))
        
        # Compute the increments of Brownian motion: dW = sqrt(dt) * Z.
        dW = np.sqrt(dG) * Z

        # Solve the martingale equation for the selected EMM
        if emm == "mean-correcting":
            # For the exact solution, the multiplicative increment for each time step is:
            incr_exact = np.exp((r - q + np.log(1 - m *kappa -0.5 * delta*delta*kappa)/kappa) * dt + m * dG + delta * dW)
            # The exact solution is obtained via cumulative product:
            S_exact = S0 * np.concatenate((np.ones((num_paths, 1)), np.cumprod(incr_exact, axis=1)), axis=1)
    
            # For the Euler approximation, the update is:
            incr_euler = 1 + (r - q + np.log(1 - m *kappa -0.5 * delta*delta*kappa)/kappa) * dt + m * dG + delta * dW
            S_euler = S0 * np.concatenate((np.ones((num_paths, 1)), np.cumprod(incr_euler, axis=1)), axis=1)
        elif emm == "Esscher":
            raise FeatureNotImplementedError(emm)
        else:
            raise UnsupportedEMMTypeError(emm)
    
    
        # Create a time array from 0 to T with (num_timesteps+1) points
        time = np.linspace(0, T, num_timesteps + 1)
    
        paths = {"time": time, "S_exact": S_exact, "S_euler": S_euler}

        return paths
