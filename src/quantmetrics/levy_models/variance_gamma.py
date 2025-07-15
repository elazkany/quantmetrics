# quantmetrics/levy_models/variance_gamma.py
from .levy_model import LevyModel
import numpy as np
from scipy.optimize import minimize, brute
import scipy.stats as st
import time
import math
from typing import Optional
from scipy.special import gamma, kn


class VarianceGamma(LevyModel):
    """
    Variance Gamma model.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    mu : float
        Expected return (drift).
    m : float
        Drift of the subordinated Brownian motion at a random time given by a Gamma process. If m = 0, then we have a symmetric variance gamma distribution.
    delta : float
        Volatility of the subordinated Brownian motion at a random time given by a Gamma process.
    kappa : float
        Variance rate of the subordinator Gamma process.
    """
    def __init__(
        self,
        S0: float = 50,
        mu: float = 0.05,
        m : float = -0.1,
        delta: float = 0.2,
        kappa: float = 0.1,
    ):
        

        self.S0 = S0
        self._mu = mu
        self._m = m
        self._delta = delta
        self._kappa = kappa

        self.params = {
            "S0": self.S0,
            "mu": self._mu,
            "m": self._m,
            "delta": self._delta,
            "kappa": self._kappa,
        }

        self.model_params = {
            "mu": self._mu,
            "m": self._m,
            "delta": self._delta,
            "kappa": self._kappa,
        }

        #super().__init__(self.params)
        
        
    @property
    def mu(self) -> float:
        return self._mu
    
    @mu.setter
    def mu(self, value: float):
        self._mu = value
        self.params['mu'] = value
        self.model_params['mu'] = value
        
        
    @property
    def m(self) -> float:
        return self._m
    
    @m.setter
    def m(self, value: float):
        self._m = value
        self.params['m'] = value
        self.model_params['m'] = value

    @property
    def delta(self) -> float:
        return self._delta
    
    @delta.setter
    def delta(self, value: float):
        self._delta = value
        self.params['delta'] = value
        self.model_params['delta'] = value

    @property
    def kappa(self) -> float:
        return self._kappa
    
    @kappa.setter
    def kappa(self, value: float):
        self._kappa = value
        self.params['kappa'] = value
        self.model_params['kappa'] = value

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
            self.model_params['delta'] > 0.0,
            self.model_params['kappa'] > 0.0,
            ])

    def pdf(self, data: np.ndarray, est_params: np.ndarray) -> np.ndarray:
        """
        Probability density function for the Variance Gamma model.

        Parameters
        ----------
        data : np.ndarray
            The data points for which the PDF is calculated.
        est_params : np.ndarray
            Estimated parameters (mu, m, delta, kappa).

        Returns
        -------
        np.ndarray
            The probability density values.
        """
        mu, m, delta, kappa = est_params
        if not self.model_params_conds_valid: 
            return 500.0
        
        x = data - mu - np.log(1 - m * kappa - delta**2 * kappa /2)/kappa

        Bessel_order = 1/kappa - 0.5
        Bessel_arg = (x**2 * (2*delta**2/kappa + m**2))**(0.5)/(delta**2)

        vgp_pdf = ( (2*np.exp(m*x/(delta**2))) / (kappa**(1/kappa) * (2*np.pi)**0.5 * delta * gamma(1/kappa) ) ) * (x**2 / (2*delta**2/kappa + m**2) )**(1/(2*kappa) -0.25) * kn(Bessel_order, Bessel_arg) 
        return vgp_pdf

    def fit(
        self,
        data: np.ndarray,
        method: str = "Nelder-Mead",
        init_params: Optional[np.ndarray] = None,
        brute_tuple: tuple = (
            (-1, 1, 0.5),  # mu
            (-1, 1, 0.5),  # m
            (0.05, 2.05, 0.5),  # delta
            (0.3, 0.5, 0.1),  # kappa
        ),
    ):
        """
        Fit the variance Gamma model to the data using Maximum Likelihood Estimation (MLE).

        Parameters
        ----------
        data : np.ndarray
            The data points to fit the model.

        method : str
            The minimization method, defualt is "Nelder-Mead". Other options are the same as for the minimize function from scipy.optimize.

        init_params : np.ndarray
            A 5x1-dimensional numpy array containing the initial estimates for the drift (mu) and volatility (sigma).

        brute_tuple : tuple
            If initial parameters are not specified, the brute function is applied with a 5x3-dimensional tuple for each parameter as (start value, end value, step size).

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
        #print(f"Elapsed time is {end_time - start_time} seconds")

        return result
