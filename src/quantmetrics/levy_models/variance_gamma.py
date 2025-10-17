# quantmetrics/levy_models/variance_gamma.py
from .levy_model_base import LevyModel
import numpy as np
from scipy.optimize import minimize, differential_evolution
import scipy.special as spsp
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
    def model_params_conds_valid(self) -> bool:
        return self._delta > 0.0 and self._kappa > 0.0

    def logpdf(self, data: np.ndarray, est_params: np.ndarray) -> np.ndarray:
        """
        Numerically stable log‐pdf of the VG distribution.
        Returns -inf for any parameter set that violates domain constraints.
        """
        mu, m, delta, kappa = est_params

        # enforce positivity
        if delta <= 0 or kappa <= 0:
            return np.full_like(data, -np.inf)

        # shift to enforce E[e^X] = 1
        arg_inv = 1.0 - m * kappa - 0.5 * delta**2 * kappa
        if arg_inv <= 0:
            return np.full_like(data, -np.inf)

        shift = np.log(arg_inv) / kappa
        x = data - mu - shift

        # precompute constants
        beta = 2.0 * delta**2 / kappa + m**2
        order = 1.0 / kappa - 0.5
        arg_bessel = np.abs(x) * np.sqrt(beta) / (delta**2)

        # avoid invalid Bessel arguments
        with np.errstate(divide="ignore", invalid="ignore"):
            log_bess = np.log(spsp.kv(order, arg_bessel))
        log_bess = np.where(np.isfinite(log_bess), log_bess, -np.inf)

        # log‐coefficients
        log_coef = (
            np.log(2.0)
            - (1.0 / kappa) * np.log(kappa)
            - 0.5 * np.log(2.0 * np.pi)
            - np.log(delta)
            - spsp.loggamma(1.0 / kappa)
        )

        # exponent term
        log_exp = m * x / (delta**2)

        # power term: (x^2 / beta)^(1/(2κ) − 1/4)
        # => (1/(2κ) − 1/4) * [2 log|x| − log(beta)]
        coeff_pow = 1.0 / (2.0 * kappa) - 0.25
        with np.errstate(divide="ignore"):
            log_pow = coeff_pow * (2.0 * np.log(np.abs(x)) - np.log(beta))

        # assemble logpdf
        logpdf_vals = log_coef + log_exp + log_pow + log_bess

        # replace any NaN with -inf so optimizer knows to avoid
        return np.where(np.isfinite(logpdf_vals), logpdf_vals, -np.inf)
    

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
        return np.exp(self.logpdf(data, est_params))

    def _moment_init(self, data: np.ndarray) -> np.ndarray:
        """
        Empirical‐moment initial guess:
          μ0 = mean(data)
          m0 = 0
          δ0 = std(data)/2
          κ0 = 1
        """
        mu0 = np.mean(data)
        m0 = 0.0
        delta0 = max(np.std(data) / 2.0, 1e-6)
        kappa0 = 1.0
        return np.array([mu0, m0, delta0, kappa0])


    def fit(
        self,
        data: np.ndarray,
        init_params: Optional[np.ndarray] = None,
        bounds: Optional[list] = None,
        max_global_iter: int = 50,
        multi_start: int = 3,
    ):
        """
        Fit the variance Gamma model to the data using Maximum Likelihood Estimation (MLE).

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

        def neg_ll(params):
            ll = self.logpdf(data, params)
            # if ANY logpdf is -inf => likelihood zero => penalize heavily
            if np.any(ll == -np.inf):
                return 1e20
            return -np.sum(ll)

        # default bounds: mu, m unconstrained; delta>0; kappa>0
        if bounds is None:
            bounds = [
                (None, None),  # mu
                (None, None),  # m
                (1e-6, None),  # delta
                (1e-6, None),  # kappa
            ]

        # 1) initial guess
        if init_params is None:
            x0 = self._moment_init(data)
            # 2) global search
            de = differential_evolution(
                neg_ll, bounds=bounds, maxiter=max_global_iter, polish=False
            )
            x0 = de.x
        else:
            x0 = init_params

        # 3) local polish
        best = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)

        # 4) multi‐start
        for _ in range(multi_start):
            trial = x0 + 0.1 * np.random.randn(4)
            res = minimize(neg_ll, trial, method="L-BFGS-B", bounds=bounds)
            if res.fun < best.fun:
                best = res

        # store results
        self._mu, self._m, self._delta, self._kappa = best.x
        self.params.update(mu=self._mu, m=self._m, delta=self._delta, kappa=self._kappa)
        self.model_params.update(self.params)
        return best
    
    def levy_density(self, x, eps_levy = 1e-16):
        m = self._m
        delta = self._delta
        kappa = self._kappa
        # Precompute reused constants for speed
        delta2 = delta * delta
        ax = np.abs(x)

        # The following is to avoid division by zero
        ax_safe = np.where(ax < eps_levy, eps_levy, ax)

        density_func = np.exp(
            m * x / delta2 
            - ax * np.sqrt(m**2 + 2 * delta2 / kappa) / delta2
            ) / (kappa * ax_safe)
    
        return density_func
