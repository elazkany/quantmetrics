#src\quantmetrics\price_calculators\gbm_pricing\gbm_calculator.py

import numpy as np
from typing import TYPE_CHECKING, Union, Dict

from quantmetrics.price_calculators.base_calculator import BaseCalculator
from quantmetrics.price_calculators.gbm_pricing.gbm_closed_form import GBMClosedForm
from quantmetrics.price_calculators.gbm_pricing.gbm_characteristic_function import GBMCharacteristicFunction
from quantmetrics.price_calculators.gbm_pricing.gbm_paths_Q import GBMSimulatePathsQ

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option


class GBMCalculator(BaseCalculator):
    """
    Calculator for option pricing under the Geometric Brownian Motion (GBM) model.

    This class provides a unified interface for computing option prices and related quantities
    using the GBM model. It leverages several specialized classes to perform tasks such as 
    computing the closed-form solution for European options, evaluating the characteristic 
    function (useful for Fourier-based methods), and simulating asset price paths under GBM dynamics.

    Parameters
    ----------
    model : LevyModel
        An instance of a Levy model that defines the underlying asset dynamics, including parameters
        such as the initial asset price (S0) and volatility (sigma).
    option : Option
        An instance of an Option containing parameters such as the risk-free rate (r), dividend yield (q),
        time to maturity (T), strike price (K), and payoff type.

    Methods
    -------
    calculate_closed_form() : Union[float, np.ndarray]
        Computes the option price using the closed-form solution for the GBM model.
    calculate_characteristic_function(u : np.ndarray) : np.ndarray
        Evaluates the characteristic function of the GBM model at the provided input values.
        This is typically used in Fourier-based pricing methods.
    simulate_paths_Q(num_timesteps : int, num_paths : int, seed : int) : dict
        Simulates asset price paths of the GBM dynamics the risk-neutral measure :math:`\\mathbb{Q}`.
        Returns a dictionary with keys:
        
        - 'time': 1D array of time points.
        - 'S': The exact solution paths (via the multiplicative formula).
        - 'S_euler': The Euler–Maruyama approximation paths.
    """
    def __init__(self, model: "LevyModel", option: "Option"):
        super().__init__(model, option)

    def calculate_closed_form(self) -> Union[float, np.ndarray]:
        return GBMClosedForm(self.model, self.option).calculate()

    def calculate_characteristic_function(
            self,
            u: np.ndarray,
            exact = False,
            theta = None,
            L=1e-12,
            M=1.0,
            N_center=150,
            N_tails=100,
            EXP_CLIP=700,
            search_bounds = (-50, 50),
            xtol=1e-8,
            rtol=1e-8,
            maxiter=500,
            M_int = 100,
            N_int = 10_000,
            sanity_theta: float = 1.0,
            chunk_u = None,
            ) -> np.ndarray:
        return GBMCharacteristicFunction(self.model, self.option)(
            u=u,
            exact=exact,
            theta=theta,
            L=L,
            M=M,
            N_center=N_center,
            N_tails=N_tails,
            EXP_CLIP=EXP_CLIP,
            search_bounds=search_bounds,
            xtol=xtol,
            rtol=rtol,
            maxiter=maxiter,
            M_int=M_int,
            N_int=N_int,
            sanity_theta=sanity_theta,
            chunk_u = chunk_u,)

    def simulate_paths_Q(self, num_timesteps: int, num_paths: int, seed: int) -> Dict[str, np.ndarray]:
        return GBMSimulatePathsQ(self.model, self.option).simulate(num_timesteps=num_timesteps, num_paths=num_paths, seed=seed)
