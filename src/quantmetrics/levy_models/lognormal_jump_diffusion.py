# quantmetrics/levy_models/lognormal_jump_diffusion.py
from .levy_model_base import LevyModel
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import factorial
import scipy.stats as st
import time
import math
from typing import Optional


class LognormalJumpDiffusion(LevyModel):
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
    ``lambda_`` : float
        Jump intensity rate is strictly greater than zero.
    muJ : float
        Mean jump size is strictly greater than -1 and non-zero.
    sigmaJ : float
        Standard deviation of the jump size. It is a positive number.
    N : int
        Number of big jumps (the Poisson jumps).
    """
    def __init__(
        self,
        S0: float = 50,
        mu: float = 0.05,
        sigma: float = 0.2,
        lambda_: float = 1,
        muJ: float = -0.1,
        sigmaJ: float = 0.1,
        N: int = 10,
    ):
        

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
    def model_params_conds_valid(self) -> bool:
        return (
            self.model_params["sigma"] > 0.0
            and self.model_params["lambda_"] > 0.0
            and self.model_params["sigmaJ"] > 0.0
        )

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
        if sigma <= 0 or lambda_ <= 0 or sigmaJ <= 0:
            return np.zeros_like(data) + 1e-300

        drift = mu - 0.5 * sigma ** 2 - lambda_ * (np.exp(muJ + 0.5 * sigmaJ ** 2) - 1)

        n = np.arange(self.N + 1)
        pmf_n = np.exp(-lambda_) * lambda_ ** n / factorial(n)
        means = drift + n * muJ
        stds = np.sqrt(sigma ** 2 + n * sigmaJ ** 2)

        pdf_matrix = st.norm.pdf(data[:, None], means[None, :], stds[None, :])
        return np.dot(pdf_matrix, pmf_n)
    
    def _moment_init(self, data: np.ndarray) -> np.ndarray:
        """
        Build an initial guess from empirical moments of the data.
        """
        mu0 = np.mean(data)
        sigma0 = np.std(data)
        lambda0 = 0.5
        muJ0 = np.median(data) - mu0
        sigmaJ0 = max(sigma0 / 2, 1e-3)
        return np.array([mu0, sigma0, lambda0, muJ0, sigmaJ0])


    def fit(
        self,
        data: np.ndarray,
        init_params: Optional[np.ndarray] = None,
        bounds: Optional[list] = None,
        max_global_iter: int = 50,
        multi_start: int = 3,
    ):
        """
        Fit the constant jump-diffusion model to the data using Maximum Likelihood Estimation (MLE).

        Parameters
        ----------
        data : np.ndarray
            The data points to fit the model.

        init_params : np.ndarray
            A 5x1-dimensional numpy array containing the initial estimates for the drift (mu) and volatility (sigma).

        Returns
        -------
        minimize
            The result of the minimization process containing the estimated parameters.
        """

        def neg_log_likelihood(params):
            p = self.pdf(data, params)
            return -np.sum(np.log(np.clip(p, 1e-300, None)))

        if bounds is None:
            bounds = [
                (None, None),
                (1e-6, None),
                (1e-6, None),
                (None, None),
                (1e-6, None),
            ]

        # 1) initial guess
        if init_params is None:
            x0 = self._moment_init(data)
            # 2) global search
            de_res = differential_evolution(
                neg_log_likelihood,
                bounds=bounds,
                maxiter=max_global_iter,
                polish=False,
            )
            x0 = de_res.x
        else:
            x0 = init_params

        # 3) local polish with bounds
        best = minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
        )

        # 4) multi-start refinements
        for _ in range(multi_start):
            trial = x0 + 0.1 * np.random.randn(5)
            res = minimize(
                neg_log_likelihood,
                trial,
                method="L-BFGS-B",
                bounds=bounds,
            )
            if res.fun < best.fun:
                best = res

        # store and return
        self._mu, self._sigma, self._lambda_, self._muJ, self._sigmaJ = best.x
        self.params.update(
            mu=self._mu,
            sigma=self._sigma,
            lambda_=self._lambda_,
            muJ=self._muJ,
            sigmaJ=self._sigmaJ,
        )
        self.model_params.update(self.params)
        return best
    
    def levy_density(self, x):
        sigmaJ2 = self._sigmaJ * self._sigmaJ
        muJ = self._muJ
        xs = (x - muJ) * (x - muJ)
        density_func = np.exp(-0.5 * xs/sigmaJ2) / np.sqrt(2 * np.pi * sigmaJ2)
        return density_func