from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING, Union, Dict

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option


class BaseCalculator(ABC):
    def __init__(self, model: "LevyModel", option: "Option"):
        self.model = model
        self.option = option

    def _validate_pricing_params():
        pass

    @abstractmethod
    def calculate_closed_form(self) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def calculate_characteristic_function(self, u: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def simulate_paths_Q(self, num_timesteps: int, num_paths: int , seed: int ) -> Dict[str, np.ndarray]:
        pass