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

    def calculate_characteristic_function(self, u: np.ndarray) -> np.ndarray:
        return VGCharacteristicFunction(self.model, self.option).calculate(u=u)

    def simulate_paths_Q(self, num_timesteps: int, num_paths: int, seed: int) -> float:
        return VGSimulatePathsQ(self.model, self.option).simulate(num_timesteps=num_timesteps, num_paths=num_paths, seed=seed)
