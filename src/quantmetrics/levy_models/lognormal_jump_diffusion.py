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
        sigmaJ: float = 0.04625031,
        N: int = 10,
    ):
        """
        Lognormal jump-diffusion model.

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
        muJ : float
            Mean jump size is strictly greater than -1 and non-zero.
        sigmaJ : float
            Standard deviation of the jump size. It is a positive number.
        N : int
            Number of big jumps (the Poisson jumps).
        """

        self.S0 = S0
        self._mu = mu
        self._sigma = sigma
        self._lambda_ = lambda_
        self._muJ = muJ
        self._sigmaJ = sigmaJ
        self.N = N

        self.params = {
            "S0": self.S0,
            "mu": self._mu,
            "sigma": self._sigma,
            "lambda_": self._lambda_,
            "muJ": self._muJ,
            "sigmaJ": self._sigmaJ,
            "N": self.N,
        }

        self.model_params = {
            "mu": self._mu,
            "sigma": self._sigma,
            "lambda_": self._lambda_,
            "muJ": self._muJ,
            "sigmaJ": self._sigmaJ,
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
    def lambda_(self) -> float:
        return self._lambda_
    
    @lambda_.setter
    def lambda_(self, value: float):
        self._lambda_ = value
        self.params['lambda_'] = value
        self.model_params['lambda_'] = value

    @property
    def muJ(self) -> float:
        return self._muJ
    
    @muJ.setter
    def muJ(self, value: float):
        self._muJ = value
        self.params['muJ'] = value
        self.model_params['muJ'] = value

    @property
    def sigmaJ(self) -> float:
        return self._sigmaJ
    
    @sigmaJ.setter
    def sigmaJ(self, value: float):
        self._sigmaJ = value
        self.params['sigmaJ'] = value
        self.model_params['sigmaJ'] = value

    @property
    def model_params_conds_valid(self):
        return self._validate_model_params()
    
    def _validate_model_params(self) -> bool:
        """
        Validate model parameters

        Returns
        -------
        bool
            True if all conditions on parameters are met, False otherwise.
        """
        return all([
            self.model_params['sigma'] > 0.0,
            self.model_params['lambda_'] > 0.0,
            self.model_params['sigmaJ'] > 0.0,
            ])

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
        mu, sigma, lambda_, muJ, sigmaJ = est_params
        if not self.model_params_conds_valid: #sigma <= 0.0 or lambda_ <= 0.0 or sigmaJ <= 0.0:
            return 500.0
        else:
            drift = mu - 0.5 * sigma**2  # TODO: consider a different drift

        sum_n = 0.0
        for n in range(0, self.N + 1):
            mean_n = drift + n * muJ
            std_n = np.sqrt(sigma**2 + n * sigmaJ**2)
            poi_pmf = np.exp(-lambda_) * lambda_**n / math.factorial(n)
            sum_n = sum_n + poi_pmf * st.norm.pdf(data, loc=mean_n, scale=std_n)
        return sum_n

    def fit(
        self,
        data: np.ndarray,
        method: str = "Nelder-Mead",
        init_params: Optional[np.ndarray] = None,
        brute_tuple: tuple = (
            (-1, 1, 0.5),  # mu
            (0.05, 2, 0.5),  # sigma
            (0.10, 0.401, 0.1),  # lambda
            (-0.5, 1, 0.1),  # muJ
            (0.05, 5, 0.5),  # sigmaJ
        ),
    ):
        """
        Fit the constant jump-diffusion model to the data using Maximum Likelihood Estimation (MLE).

        Parameters
        ----------
        data : np.ndarray
            The data points to fit the model.

        method : str
            The minimization method, defualt is "Nelder-Mead". Other options are the same as for the minimize function from scipy.optimize.

        init_params : np.ndarray
            A 5x1-dimensional numpy array containing the initial estimates for the drift (mu) and volatility (sigma).

        brute_tuple : tuple
            If initial parameters are not specified, the brute function is applied with a 5x3-dimensional tuple for each parameter
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
            params = brute(MLE, brute_tuple, finish=None)
        else:
            params = init_params

        result = minimize(MLE, params, method=method)

        end_time = time.time()
        print(f"Elapsed time is {end_time - start_time} seconds")

        return result
