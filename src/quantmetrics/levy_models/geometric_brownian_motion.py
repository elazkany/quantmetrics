# levy_models/geometric_brownian_motion.py
from .levy_model import LevyModel
import numpy as np
from scipy.optimize import minimize, brute
import scipy.stats as st
import time


class GeometricBrownianMotion(LevyModel):
    def __init__(
        self,
        S0: float = 60,
        mu: float = 0.025,
        sigma: float = 0.02320006,
    ):
        """
        Geometric Brownian motion model.

        Parameters
        ----------
        S0 : float
            Initial stock price.
        mu : float
            Expected return (drift).
        sigma : float
            Volatility (annualized). Divide by the square root of the number of days in a year (e.g., 360) to convert to daily.
        """

        params = {
            "S0": S0,
            "mu": mu,
            "sigma": sigma,
        }
        super().__init__(params)
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma

    def pdf(self, data: np.ndarray, est_params: np.ndarray) -> np.ndarray:
        """
        Probability density function for the Geometric Brownian Motion model.

        Parameters
        ----------
        data : np.ndarray
            The data points for which the PDF is calculated.
        est_params : np.ndarray
            Estimated parameters (mu, sigma).

        Returns
        -------
        np.ndarray
            The probability density values.
        """
        mu, sigma = est_params
        if sigma <= 0.0:
            return 500.0
        else:
            drift = mu - 0.5 * sigma**2
            return st.norm.pdf(data, loc=drift, scale=sigma)

    def fit(self, data: np.ndarray, method : str = "Nelder-Mead", init_params : np.ndarray = None, brute_tuple : tuple = ((-1,1,0.5),(0.05,2,0.5))):
        """
        Fit the Geometric Brownian Motion model to the data using Maximum Likelihood Estimation (MLE).

        Parameters
        ----------
        data : np.ndarray
            The data points to fit the model.

        method : str
            The minimization method, defualt is "Nelder-Mead".

        init_params : np.ndarray
            A 2-dimensional numpy array containing the initial estimates for the drift (mu) and volatility (sigma).

        brute_tuple : tuple
            If initial parameters are not specified, the brute function is applied with a 2-dimensional tuple for each parameter 
        as (start value, end value, step size).

        Returns
        -------
        minimize
            The result of the minimization process containing the estimated parameters.
        """

        def MLE(params):
            return -np.sum(
                np.log(
                    self.pdf(
                        data=data,
                        est_params=params,
                    )
                )
            )
        
        start_time = time.time()

        if init_params is None:
            params = brute(MLE, brute_tuple, finish = None)
        else:
            params = init_params

        result = minimize(MLE, params, method=method)

        end_time = time.time()
        print(f"Elapsed time is {end_time - start_time} seconds")
        
        return result
