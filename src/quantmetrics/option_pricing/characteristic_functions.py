#option_pricing/characteristics_functions.py

from quantmetrics.levy_models import GBM

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option

class CharacteristicFunction:
    def __init__(
            self,
            model : 'LevyModel',
            option : 'Option'):
        """
        Initialize the CharacteristicFunction with a model and an option.

        Parameters:
        model (LevyModel): The Levy model.
        option (Option): The option parameters.
        """
        self.model = model
        self.option = option

    def calculate(self, u: np.ndarray) -> np.ndarray:
        """
        Calculate the characteristic function for the given model.

        Parameters:
        u (np.ndarray): Input array for the characteristic function.

        Returns:
        np.ndarray: The characteristic function values.
        """
        if isinstance(self.model, GBM):
            return self._gbm_characteristic_function(u)

    def _gbm_characteristic_function(self, u: np.ndarray) -> np.ndarray:
        """
        Calculate the characteristic function for the GBM model.

        Parameters:
        u (np.ndarray): Input array for the characteristic function.

        Returns:
        np.ndarray: The characteristic function values.
        """
        sigma = self.model.sigma
        r = self.option.r
        T = self.option.T

        b = r - 0.5 * sigma**2
        char_func = np.exp(T * (-0.5 * sigma**2 * u**2 + 1j * u * b))
        return char_func
