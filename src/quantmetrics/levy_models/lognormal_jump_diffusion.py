# quantmetrics/levy_models/lognormal_jump_diffusion.py
from .levy_model import LevyModel
import numpy as np
from scipy.optimize import minimize, brute
import scipy.stats as st
import time
import math
from typing import Optional

class LognormalJumpDiffusion(LevyModel):
    def __init__(
        self,
        S0: float = 60,
        mu: float = 0.00049883,
        sigma: float = 0.02320006,
        lambda_: float = 0.01508188,
        muJ: float = -0.0934457,
        sigmaJ : float = 0.04625031,
        N : int = 10,
    ):
        """
        Constant jump-diffusion model.

        Parameters
        ----------
        S0 : float
            Initial stock price.
        mu : float
            Expected return (drift).
        sigma : float
            Volatility (annualized). Divide by the square root of the number of days in a year (e.g., 360) to convert to daily.
        lambda_ : float
            Jump intensity rate is strictly greater than zero.
        gamma : float
            Mean jump size is strictly greater than -1 and non-zero.
        N : int
            Number of big jumps (the Poisson jumps).
        """

        params = {
            "S0": S0,
            "mu": mu,
            "sigma": sigma,
            "lamnda": lambda_,
            "muJ" : muJ,
            "sigmaJ" : sigmaJ,
            "N": N,
        }
        super().__init__(params)
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.lambda_ = lambda_
        self.muJ = muJ
        self.sigmaJ = sigmaJ
        self.N = N

    def pdf(self, data: np.ndarray, est_params: np.ndarray) -> np.ndarray:
        """
        Probability density function for the lognormal jump-diffusion model.

        Parameters
        ----------
        data : np.ndarray
            The data points for which the PDF is calculated.
        est_params : np.ndarray
            Estimated parameters (mu, sigma, lambda, muJ, sigmaJ).

        Returns
        -------
        np.ndarray
            The probability density values.
        """
        mu, sigma, lambda_, gamma = est_params
        if sigma <= 0.0 or lambda_ <= 0.0 or gamma ==0.0 or gamma <= -1:
            return 500.0
        else:
            drift = mu - 0.5 * sigma**2

        sum_n = 0.0
        for n in range(0, self.n + 1):
            mean_n = drift + n * gamma
            std_n = np.sqrt(sigma**2)
            poi_pmf = np.exp(-lambda_) * lambda_**n / math.factorial(n)
            sum_n = sum_n + poi_pmf * st.norm.pdf(data, loc=mean_n, scale=std_n)
        return sum_n


    def fit(self, data: np.ndarray, method : str = "Nelder-Mead", init_params : Optional[np.ndarray] = None, brute_tuple : tuple = ((-1,1,0.5),(0.05,2,0.5), (0.10,0.401,0.1), (-0.5,1,0.1))):
        """
        Fit the constant jump-diffusion model to the data using Maximum Likelihood Estimation (MLE).

        Parameters
        ----------
        data : np.ndarray
            The data points to fit the model.

        method : str
            The minimization method, defualt is "Nelder-Mead". Other options are the same as for the minimize function from scipy.optimize.

        init_params : np.ndarray
            A 2-dimensional numpy array containing the initial estimates for the drift (mu) and volatility (sigma).

        brute_tuple : tuple
            If initial parameters are not specified, the brute function is applied with a 4x3-dimensional tuple for each parameter 
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
