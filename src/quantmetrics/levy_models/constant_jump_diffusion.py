# levy_models/constant_jump_diffusion.py
from .levy_model_base import LevyModel
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import gammaln, logsumexp
import scipy.stats as st
import time
import math
from typing import Optional

class ConstantJumpDiffusion(LevyModel):
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
    ``lambda_`` : float
        Jump intensity rate is strictly greater than zero.
    gamma : float
        Mean jump size is strictly greater than -1 and non-zero.
    N : int
        Number of big jumps (the Poisson jumps).
    """
    def __init__(
        self,
        S0: float = 50,
        mu: float = 0.05,
        sigma: float = 0.2,
        lambda_: float = 1,
        gamma: float = -0.1,
        N : int = 10,
    ):
        

        self.S0 = S0
        self._mu = mu
        self._sigma = sigma
        self._lambda_ = lambda_
        self._gamma = gamma
        self.N = N

        self.params = {
            "S0": self.S0,
            "mu": self._mu,
            "sigma": self._sigma,
            "lambda_": self._lambda_,
            "gamma": self._gamma,
            "N": self.N,
        }

        self.model_params = {
            "mu": self._mu,
            "sigma": self._sigma,
            "lambda_": self._lambda_,
            "gamma": self._gamma,
        }

        super().__init__(self.params)
        

    @property
    def model_params_conds_valid(self) -> bool:
        return (
            self._sigma > 0.0
            and self._lambda_ > 0.0
            and self._gamma > -1.0
            and self._gamma != 0.0
        )
    
    def logpdf(self, data: np.ndarray, est_params: np.ndarray) -> np.ndarray:
        """
        Numerically stable log‐pdf of the constant jump‐diffusion:
          f(x) = sum_{n=0}^N Poisson(n;λ) * Normal(x; μ_n, σ)
        with μ_n = drift + n*γ
        """
        mu, sigma, lam, gamma = est_params

        # domain checks
        if sigma <= 0 or lam <= 0 or gamma <= -1 or gamma == 0:
            return np.full_like(data, -np.inf)

        # drift correction
        drift = mu - 0.5 * sigma**2 - lam * (np.exp(gamma) - 1.0)

        # precompute Poisson log‐pmf for n=0..N
        n = np.arange(self.N + 1)
        log_pmf = -lam + n * np.log(lam) - gammaln(n + 1)

        # component means and shared sigma
        means = drift + n * gamma
        log_norm_const = -0.5 * (np.log(2 * np.pi) + 2 * np.log(sigma))

        # compute log‐pdf mixture via log‐sum‐exp
        # data[:,None] - means[None,:] → shape (len(data), N+1)
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = (data[:, None] - means[None, :]) / sigma
            log_norm = log_norm_const - 0.5 * dev**2
            log_terms = log_pmf[None, :] + log_norm

        # log‐sum‐exp across jumps
        logpdf_vals = logsumexp(log_terms, axis=1)

        # replace non‐finite with -inf
        return np.where(np.isfinite(logpdf_vals), logpdf_vals, -np.inf)


    def pdf(self, data: np.ndarray, est_params: np.ndarray) -> np.ndarray:
        """
        Probability density function for the constant jump-diffusion model.

        Parameters
        ----------
        data : np.ndarray
            The data points for which the PDF is calculated.
        est_params : np.ndarray
            Estimated parameters (mu, sigma, lambda, gamma).

        Returns
        -------
        np.ndarray
            The probability density values.
        """
        return np.exp(self.logpdf(data, est_params))

    def _moment_init(self, data: np.ndarray) -> np.ndarray:
        """
        Moment‐based initial guess:
          μ0 = mean(data)
          σ0 = std(data)
          λ0 = 0.5
          γ0 = median(data) - μ0

          A good initial guess positions the optimizer near the global optimum, speeding convergence and reducing the chance of getting stuck in poor local minima.
        """
        mu0 = np.mean(data)
        sigma0 = np.std(data)
        lam0 = 0.5
        gamma0 = np.median(data) - mu0
        # clamp gamma0 inside (-0.9, 0.9)
        gamma0 = np.clip(gamma0, -0.9, 0.9)
        return np.array([mu0, sigma0, lam0, gamma0])

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
        This performs the following steps:
            1) Choose or Compute an initial guess.
            2) (Optional) Globally explore the landscape with Differential Evolution.
            3) Locally refine using L-BFGS-B under parameter bounds.
            4) Robustify by re-optimizing from multiple nearby starting points.

        Parameters
        ----------
        data : np.ndarray
            The data points to fit the model.

        init_params : np.ndarray
            A 4x1-dimensional numpy array containing the initial estimates for the drift (mu) and volatility (sigma).

        Returns
        -------
        minimize
            The result of the minimization process containing the estimated parameters.
        
        """

        def neg_ll(params):
            lp = self.logpdf(data, params)
            # any -inf in logpdf => zero likelihood => heavy penalty
            if np.any(lp == -np.inf):
                return 1e20
            return -np.sum(lp)

        # default bounds: mu free, sigma>0, lambda>0, gamma>-1
        if bounds is None:
            bounds = [
                (None, None),      # mu
                (1e-6, None),      # sigma > 0
                (1e-6, None),      # lambda > 0
                (-0.999, None),    # gamma > -1
            ]

        # 1) initial guess
        if init_params is None:
            x0 = self._moment_init(data)
            # 2) global search to explor the parameter space broadly.
            de = differential_evolution(
                neg_ll, bounds=bounds, maxiter=max_global_iter, polish=False
            )
            x0 = de.x
        else:
            x0 = init_params

        # 3) local polish
        best = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)

        # 4) multi-start Even with a good guess and global search, L-BFGS-B can still settle in a sub-optimal local minimum.
        for _ in range(multi_start):
            trial = x0 + 0.1 * np.random.randn(4) # Generate a perturbed trial point
            res = minimize(neg_ll, trial, method="L-BFGS-B", bounds=bounds) # - Run the same L-BFGS-B polish from trial.
            if res.fun < best.fun:
                # If the new solution achieves a lower negative log-likelihood, replace the current best.
                best = res

        # update parameters
        self._mu, self._sigma, self._lambda_, self._gamma = best.x
        self.params.update(
            mu=self._mu,
            sigma=self._sigma,
            lambda_=self._lambda_,
            gamma=self._gamma,
        )
        self.model_params.update(self.params)
        return best