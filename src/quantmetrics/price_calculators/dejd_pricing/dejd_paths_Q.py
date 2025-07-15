from quantmetrics.risk_calculators.martingale_equation import RiskPremium
from quantmetrics.utils.exceptions import FeatureNotImplementedError, UnsupportedEMMTypeError
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option

class DEJDSimulatePathsQ:
    def __init__(self, model: "LevyModel", option: "Option"):
        """
        Initialize the DEJDSimulatePathsQ with a model and an option.

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
        return self._dejd_simulate_paths_Q(num_timesteps, num_paths, seed)

    def _dejd_simulate_paths_Q(self, num_timesteps: int, num_paths: int, seed: int) -> np.ndarray:
        """
        Generate paths for the double-exponential jump-diffusion (LJD) model.

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
        sigma = self.model.sigma
        lambda_ = self.model.lambda_
        eta1 = self.model.eta1
        eta2 = self.model.eta2
        p = self.model.p
        r = self.option.r
        T = self.option.T
        q = self.option.q
        emm = self.option.emm
        psi = self.option.psi

        np.random.seed(seed)

        dt = T / float(num_timesteps)

        """
        # Generate normally distributed random variables with shape (num_paths, num_timesteps)
        Z = np.random.standard_normal(size=(num_paths, num_timesteps))
        
        W = np.zeros((num_paths, num_timesteps + 1 ))
        
        S_exact = np.zeros(W.shape)
        S_exact[:, 0] = S0
        S_euler = np.zeros(W.shape)
        S_euler[:, 0] = S0

        time = np.zeros(W.shape[0])

        if emm == "mean-correcting":
            lambda_Q = lambda_
            kappa = p/(eta1 - 1) - (1- p)/(eta2 + 1)
        elif emm == "Esscher":
            raise FeatureNotImplementedError(emm)
        else:
            raise UnsupportedEMMTypeError(emm)

        ZPois = np.random.poisson(lambda_Q * dt, [num_paths, num_timesteps])

        J = np.zeros([ num_paths, num_timesteps])
        # Generate a Bernoulli trial to decide upward or downward jumps
        upward = np.random.rand( num_paths, num_timesteps) < p
        # Generate positive jumps (upward) and negative jumps (downward)
        J[upward] = np.random.exponential(scale=1/eta1, size=np.sum(upward))  # Upward jumps
        J[~upward] = -np.random.exponential(scale=1/eta2, size=np.sum(~upward))  # Downward jumps

        for i in range(0, num_timesteps):
            # Making sure that samples from a normal have mean 0 and variance 1
            if num_paths > 1:
                Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
                # Making sure that samples from a normal have mean 0 and variance 1
            W[:,i + 1] = W[:,i] + np.power(dt, 0.5) * Z[:,i]

            S_exact[:,i + 1] = S_exact[:,i] * np.exp( (r - q -0.5 *sigma*sigma - lambda_Q * kappa)*dt + sigma * (W[:,i + 1] - W[:,i])+ J[:,i] * ZPois[:, i])
                    
            S_euler[:,i + 1] = S_euler[:,i] + (r-q - lambda_Q * kappa) * S_euler[:,i] * dt + sigma * S_euler[:,i] * (W[:,i + 1] - W[:,i]) +(np.exp(J[:,i])-1)* S_euler[:,i] * ZPois[:,i]

            time[i + 1] = time[i] + dt

        paths = {"time": time, "S_exact": S_exact, "S_euler": S_euler}"""

        # Generate normally distributed random variables with shape (num_paths, num_timesteps)
        Z = np.random.standard_normal(size=(num_paths, num_timesteps))
        
        # Normalize each column so that each time-step's ensemble has mean 0 and variance 1.
        Z = (Z - np.mean(Z, axis=0)) / np.std(Z, axis=0)
        
        # Compute the increments of Brownian motion: dW = sqrt(dt) * Z.
        dW = np.sqrt(dt) * Z

        # Solve the martingale equation for the selected EMM
        if emm == "mean-correcting":
            lambda_Q = lambda_
            kappa = p/(eta1 - 1) - (1- p)/(eta2 + 1)
        elif emm == "Esscher":
            raise FeatureNotImplementedError(emm)
        else:
            raise UnsupportedEMMTypeError(emm)
        

        # Generate Poisson distributed random variables with intensity lambda_Q * dt
        ZPois = np.random.poisson(lambda_Q * dt, [num_paths, num_timesteps])

        J = np.zeros([ num_paths, num_timesteps])
        # Generate a Bernoulli trial to decide upward or downward jumps
        upward = np.random.rand( num_paths, num_timesteps) < p
        # Generate positive jumps (upward) and negative jumps (downward)
        J[upward] = np.random.exponential(scale=1/eta1, size=np.sum(upward))  # Upward jumps
        J[~upward] = -np.random.exponential(scale=1/eta2, size=np.sum(~upward))  # Downward jumps

        # For the exact solution, the multiplicative increment for each time step is:
        incr_exact = np.exp((r - q - 0.5 * sigma**2 - lambda_Q * kappa) * dt + sigma * dW + J * ZPois)
        # The exact solution is obtained via cumulative product:
        S_exact = S0 * np.concatenate((np.ones((num_paths, 1)), np.cumprod(incr_exact, axis=1)), axis=1)
    
        # For the Euler approximation, the update is:
        incr_euler = 1 + (r - q - lambda_Q * kappa) * dt + sigma * dW + (np.exp(J)-1)* ZPois
        S_euler = S0 * np.concatenate((np.ones((num_paths, 1)), np.cumprod(incr_euler, axis=1)), axis=1)
    
        # Create a time array from 0 to T with (num_timesteps+1) points
        time = np.linspace(0, T, num_timesteps + 1)
    
        paths = {"time": time, "S_exact": S_exact, "S_euler": S_euler}

        return paths
    

    
