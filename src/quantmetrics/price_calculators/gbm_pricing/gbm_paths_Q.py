#src\quantmetrics\price_calculators\gbm_pricing\gbm_paths_Q.py

import numpy as np
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option

class GBMSimulatePathsQ:
    """
    Implements the paths simulation for a Geometric Brownian Motion (GBM) model under the risk-neutral measure.

    Parameters
    ----------
    model : LevyModel
        A LevyModel object specifying the underlying asset's model and its parameters.
    option : Option
        An Option object specifying the option parameters: interest rate, strike price, time to maturity, dividend yield and the equivalent martingale measure.
    """
    def __init__(self, model: "LevyModel", option: "Option"):
        self.model = model
        self.option = option

    def simulate(self, num_timesteps: int, num_paths: int, seed: int) -> Dict[str, np.ndarray]:
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
            A dictionary containing:

            - `time_steps` (np.ndarray): The simulated time steps.
            - `S_exact` (np.ndarray): The simulated GBM paths using the exact solution.
            - `S_euler` (np.ndarray): The simulated GBM paths using the Euler-Maruyama discretization.

        
        Notes
        -----
        The Euler-Maruyama discretization for the :math:`i^{th}` timestep and :math:`j^{th}` path, reads:

        .. math::

            s_{i+1, j} \\approx s_{i,j} + (r-q) s_{i,j} \\Delta t + \\sigma s_{i,j} (W_{i+1, j} - W_{i,j}),

        with :math:`\\Delta t = t_{i+1} - t_i`, for any :math:`i=1,2,\\cdots , m, \\ s_0 = S(t_0)=S_0`, :math:`j = 1,2,\\cdots, N` and  :math:`W_{i+1, j} - W_{i,j} \\sim \\mathcal{N}(0, \Delta t)`

        The GBM process has as exact solution in the time interval :math:`[t_i, t_{i+1}]`,

        .. math::

            S(t_{i+1})=S(t_i)\\exp\\left\{(r-q-\\frac{\\sigma^2}{2})\\Delta t + \\sigma [W(t_{i+1}) - W(t_i)] \\right\}
            
        where

            - :math:`S_0` is the underlying price.
            - :math:`r` is the risk-free interest rate.
            - :math:`q` is the dividend yield. 
            - :math:`\\sigma` is the volatility.
        

        Examples
        --------
        >>> from quantmetrics.levy_models import GBM
        >>> from quantmetrics.option_pricing import Option
        >>> from quantmetrics.price_calculators.gbm_pricing.gbm_paths_Q import GBMSimulatePathsQ
        >>> gbm = GBM() # S0=50, sigma=0.2
        >>> option = Option(K=np.array([20,50,80])) # r=0.05, q=0.02, T=0.5
        >>> paths = GBMSimulatePathsQ(gbm, option).simulate(num_timesteps=200, num_paths=10000,seed=42)
        >>> payoff = np.maximum(paths["S_exact"][:,-1].reshape(-1,1) - option.K, 0)
        >>> option_price = np.mean(np.exp(-option.r*option.T) * payoff, axis=0)
        >>> option_price
        array([2.99914386e+01, 3.13832975e+00, 1.25147041e-03])

        References
        ----------

        .. [1] Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of political economy, 81(3), 637-654.
        
        .. [2] Matsuda, K. (2004). Introduction to option pricing with Fourier transform: Option pricing with exponential Lévy models. Department of Economics The Graduate Center, The City University of New York, 1-241.

        .. [3] Oosterlee, C. W., & Grzelak, L. A. (2019). Mathematical modeling and computation in finance: with exercises and Python and MATLAB computer codes. World Scientific.
        """
        return self._gbm_simulate_paths_Q(num_timesteps, num_paths, seed)

    def _gbm_simulate_paths_Q(self, num_timesteps: int, num_paths: int, seed: int) -> Dict[str, np.ndarray]:
        
        np.random.seed(seed)

        S0 = self.model.S0
        sigma = self.model.sigma
        r = self.option.r
        q = self.option.q
        T = self.option.T

        dt = T / float(num_timesteps)
        """
        Z = np.random.standard_normal(
            size=(num_paths,num_timesteps)
        )  # generate standard normal random variable of size.... [t,omega]
        
        # Standard Brownian motion
        W = np.zeros((num_paths, num_timesteps + 1))
        
        # The underlying process
        S_exact = np.zeros(W.shape)
        S_exact[:, 0] = S0
        S_euler = np.zeros(W.shape)
        S_euler[:, 0] = S0
        
        time = np.zeros(W.shape[0])

        for i in range(0, num_timesteps):
            # Making sure that samples from the normal distribution have mean 0 and variance 1
            if num_paths > 1:
                Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
                W[:, i + 1] = W[:,i] + np.power(dt, 0.5) * Z[:,i]
            
                S_exact[:,i + 1] = S[:,i] * np.exp( (r - q -0.5 *sigma*sigma)*dt + sigma * (W[:,i + 1] - W[:,i]))
                    
                S_euler[:,i + 1] = S_euler[:,i] + (r-q) * S_euler[:,i] * dt + sigma * S_euler[:,i] * (W[:,i + 1] - W[:,i])
                    
                time[i + 1] = time[i] + dt

        paths = {"time": time, "S_exact": S_exact, "S_euler": S_euler}
        """
        # Generate normally distributed random variables with shape (num_paths, num_timesteps)
        Z = np.random.standard_normal(size=(num_paths, num_timesteps))
        # Normalize each column so that each time-step's ensemble has mean 0 and variance 1.
        Z = (Z - np.mean(Z, axis=0)) / np.std(Z, axis=0)
    
        # Compute the increments of Brownian motion: dW = sqrt(dt) * Z.
        dW = np.sqrt(dt) * Z
    
        # For the exact solution, the multiplicative increment for each time step is:
        # exp((r - q - 0.5*sigma^2)*dt + sigma*dW)
        incr_exact = np.exp((r - q - 0.5 * sigma**2) * dt + sigma * dW)
        # The exact solution is obtained via cumulative product:
        S_exact = S0 * np.concatenate((np.ones((num_paths, 1)), np.cumprod(incr_exact, axis=1)), axis=1)
    
        # For the Euler approximation, the update is:
        # S_euler[i+1] = S_euler[i] * [1 + (r - q) * dt + sigma*dW]
        incr_euler = 1 + (r - q) * dt + sigma * dW
        S_euler = S0 * np.concatenate((np.ones((num_paths, 1)), np.cumprod(incr_euler, axis=1)), axis=1)
    
        # Create a time array from 0 to T with (num_timesteps+1) points
        time = np.linspace(0, T, num_timesteps + 1)
    
        paths = {"time": time, "S_exact": S_exact, "S_euler": S_euler}
        return paths