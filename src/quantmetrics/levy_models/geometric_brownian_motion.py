# levy_models/geometric_brownian_motion.py
from .levy_model import LevyModel
import numpy as np
from scipy.optimize import minimize, brute
import scipy.stats as st
import time


class GeometricBrownianMotion(LevyModel):
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
    def __init__(
        self,
        S0: float = 50,
        mu: float = 0.05,
        sigma: float = 0.2,
    ):

        self.S0 = S0
        self._mu = mu
        self._sigma = sigma

        self.params = {
            "S0": self.S0,
            "mu": self._mu,
            "sigma": self._sigma,
        }

        self.model_params = {
            "mu": self._mu,
            "sigma" : self._sigma,
        }        

        super().__init__(self.params)

    @property
    def mu(self) -> float:
        return self._mu
    
    @mu.setter
    def mu(self, value: float):
        self._mu = value
        self.params['mu'] = value
        self.model_params['mu'] = value
        
        
    @property
    def sigma(self) -> float:
        return self._sigma
    
    @sigma.setter
    def sigma(self, value: float):
        self._sigma = value
        self.params['sigma'] = value
        self.model_params['sigma'] = value

    @property
    def model_params_conds_valid(self):
        return self.model_params['sigma'] > 0.0
        

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
        if not self.model_params_conds_valid: #sigma <= 0.0:
            return 500.0
        else:
            drift = mu - 0.5 * sigma**2
            return st.norm.pdf(data, loc=drift, scale=sigma)

    def fit(
        self,
        data: np.ndarray,
        method: str = "Nelder-Mead",
        init_params: np.ndarray = None,
        brute_tuple: tuple = ((-1, 1, 0.5), (0.05, 2, 0.5)),
    ):
        """
        Fit the Geometric Brownian Motion model to the data using Maximum Likelihood Estimation (MLE).

        Parameters
        ----------
        data : np.ndarray
            The data points to fit the model.

        method : str
            The minimization method, defualt is "Nelder-Mead".

        init_params : np.ndarray
            A 2x1-dimensional numpy array containing the initial estimates for the drift (mu) and volatility (sigma).

        brute_tuple : tuple
            If initial parameters are not specified, the brute function is applied with a 2x3-dimensional tuple for each parameter as (start value, end value, step size).

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
            params = brute(MLE, brute_tuple, finish=None)
        else:
            params = init_params

        result = minimize(MLE, params, method=method)

        end_time = time.time()
        print(f"Elapsed time is {end_time - start_time} seconds")

        return result
