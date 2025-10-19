from quantmetrics.price_calculators.base_calculator import BaseCalculator
from quantmetrics.price_calculators.vg_pricing.vg_closed_form import VGClosedForm
from quantmetrics.price_calculators.vg_pricing.vg_characteristic_function import VGCharacteristicFunction
from quantmetrics.price_calculators.vg_pricing.vg_paths_Q import VGSimulatePathsQ

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option


class VGCalculator(BaseCalculator):
    def __init__(self, model: "LevyModel", option: "Option"):
        self.model = model
        self.option = option

    def calculate_closed_form(self) -> float:
        return VGClosedForm(self.model, self.option).calculate()

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
        return VGCharacteristicFunction(self.model, self.option)(
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
            chunk_u = chunk_u,
            )

    def simulate_paths_Q(self, num_timesteps: int, num_paths: int, seed: int) -> float:
        return VGSimulatePathsQ(self.model, self.option).simulate(num_timesteps=num_timesteps, num_paths=num_paths, seed=seed)
