# levy_models/geometric_brownian_motion.py
from .levy_model_base import LevyModel
import numpy as np
from scipy.optimize import minimize, differential_evolution
import scipy.stats as st
from typing import Optional


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
    def model_params_conds_valid(self) -> bool:
        return self._sigma > 0.0

    def logpdf(
        self,
        data: np.ndarray,
        est_params: np.ndarray
        ) -> np.ndarray:
        """
        Numerically stable log‐pdf of GBM returns:
          X ~ Normal(loc = mu - ½σ², scale=σ)
        """
        mu, sigma = est_params
        if sigma <= 0:
            return np.full_like(data, -np.inf)

        drift = mu - 0.5 * sigma**2
        return st.norm.logpdf(data, loc=drift, scale=sigma)
        

    def pdf(
        self,
        data: np.ndarray,
        est_params: np.ndarray
        ) -> np.ndarray:
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
        return np.exp(self.logpdf(data, est_params))

    def _moment_init(self, data: np.ndarray) -> np.ndarray:
        """
        Moment‐based initial guess:
          mu0    = mean(data)
          sigma0 = std(data)
        """
        mu0 = np.mean(data)
        sigma0 = max(np.std(data), 1e-6)
        return np.array([mu0, sigma0])

    def fit(
        self,
        data: np.ndarray,
        init_params: Optional[np.ndarray] = None,
        bounds: Optional[list] = None,
        max_global_iter: int = 50,
        multi_start: int = 3,
    ):
        """
        Fit the Geometric Brownian Motion model to the data using Maximum Likelihood Estimation (MLE).
        The following steps are performed:
            1) choose init (user or moment)
            2) global DE search if no user init
            3) local L-BFGS-B polish with bounds
            4) multi-start refinements

        Parameters
        ----------
        data : np.ndarray
            The data points to fit the model.

        init_params : np.ndarray
            A 2x1-dimensional numpy array containing the initial estimates for the drift (mu) and volatility (sigma).

        Returns
        -------
        minimize
            The result of the minimization process containing the estimated parameters.
        
        """

        def neg_ll(params):
            lp = self.logpdf(data, params)
            # any -inf => zero likelihood => huge penalty
            if np.any(lp == -np.inf):
                return 1e20
            return -np.sum(lp)

        # default bounds: mu free, sigma>0
        if bounds is None:
            bounds = [(None, None), (1e-6, None)]

        # 1) initial guess
        if init_params is None:
            x0 = self._moment_init(data)
            # 2) global search
            de = differential_evolution(
                neg_ll,
                bounds=bounds,
                maxiter=max_global_iter,
                polish=False,
            )
            x0 = de.x
        else:
            x0 = init_params

        # 3) local polish
        best = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)

        # 4) multi-start refinements
        for _ in range(multi_start):
            trial = x0 + 0.1 * np.random.randn(2)
            res = minimize(neg_ll, trial, method="L-BFGS-B", bounds=bounds)
            if res.fun < best.fun:
                best = res

        # store and return
        self._mu, self._sigma = best.x
        self.params.update(mu=self._mu, sigma=self._sigma)
        self.model_params.update(self.params)
        return best