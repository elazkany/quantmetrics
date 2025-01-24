# quantmetrics/levy_models/double_exponential_jump_diffusion.py
from .levy_model import LevyModel
import numpy as np
from scipy.optimize import minimize, brute
import scipy.stats as st
import time
import math
from typing import Optional


class DoubleExponentialJumpDiffusion(LevyModel):
    def __init__(
        self,
        S0: float = 100,
        mu: float = 0.025,
        sigma: float = 0.16,
        lambda_: float = 1,
        eta1: float = 10,
        eta2: float = 5,
        p: float = 0.4,
        N: int = 10,
    ):
        """
        Double exponential jump-diffusion model.

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
        eta1 : float
            ...
        eta2 : float
            ...
        p : float
            ...
        N : int
            Number of big jumps (the Poisson jumps).


        References
        ----------
        Kou, S. G. (2002). A jump-diffusion model for option pricing. Management science, 48(8), 1086-1101.
        Ramezani, C. A., & Zeng, Y. (2007). Maximum likelihood estimation of the double exponential jump-diffusion process. Annals of Finance, 3, 487-507.

        Examples
        --------
        ```python
        from quantmetrics.levy_models import DEJD
        from quantmetrics.option_pricing import Option, OptionPricer

        dejd = DEJD()
        option = Option(T=0.5, r=0.05, K=98)
        dejd_price = OptionPricer(model=dejd, option=option)
        # calculating the option price as in Kou, S. G. (2002)
        dejd_price.exact()
        ```
        """

        self.S0 = S0
        self._mu = mu
        self._sigma = sigma
        self._lambda_ = lambda_
        self._eta1 = eta1
        self._eta2 = eta2
        self._p = p
        self.N = N

        self.params = {
            "S0": self.S0,
            "mu": self._mu,
            "sigma": self._sigma,
            "lambda_": self._lambda_,
            "eta1": self._eta1,
            "eta2": self._eta2,
            "p": self._p,
            "N": self.N,
        }

        self.model_params = {
            "mu": self._mu,
            "sigma": self._sigma,
            "lambda_": self._lambda_,
            "eta1": self._eta1,
            "eta2": self._eta2,
            "p": self._p,
        }

        super().__init__(self.params)

    @property
    def mu(self) -> float:
        return self._mu

    @mu.setter
    def mu(self, value: float):
        self._mu = value
        self.params["mu"] = value
        self.model_params["mu"] = value

    @property
    def sigma(self) -> float:
        return self._sigma

    @sigma.setter
    def sigma(self, value: float):
        self._sigma = value
        self.params["sigma"] = value
        self.model_params["sigma"] = value

    @property
    def lambda_(self) -> float:
        return self._lambda_

    @lambda_.setter
    def lambda_(self, value: float):
        self._lambda_ = value
        self.params["lambda_"] = value
        self.model_params["lambda_"] = value

    @property
    def eta1(self) -> float:
        return self._eta1

    @eta1.setter
    def eta1(self, value: float):
        self._eta1 = value
        self.params["eta1"] = value
        self.model_params["eta1"] = value

    @property
    def eta2(self) -> float:
        return self._eta2

    @eta2.setter
    def eta2(self, value: float):
        self._eta2 = value
        self.params["eta2"] = value
        self.model_params["eta2"] = value

    @property
    def p(self) -> float:
        return self._p

    @p.setter
    def p(self, value: float):
        self._p = value
        self.params["p"] = value
        self.model_params["p"] = value

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
        return all(
            [
                self.model_params["sigma"] > 0.0,
                self.model_params["lambda_"] > 0.0,
                self.model_params["eta1"] > 1.0,
                self.model_params["eta2"] > 0.0,
                self.model_params["p"] >= 0.0,
                self.model_params["p"] <= 1.0,
            ]
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
        # TODO:

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
