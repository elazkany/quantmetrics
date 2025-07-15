
import numpy as np
from typing import TYPE_CHECKING

from quantmetrics.utils.exceptions import FeatureNotImplementedError, UnsupportedEMMTypeError

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option

class VGCharacteristicFunction:
    def __init__(self, model: "LevyModel", option: "Option"):
        """
        Initialize the VGCharacteristicFunction with a model and an option.

        Parameters
        ----------
        model : LevyModel
            The Levy model used for calculating the characteristic function.
        option : Option
            The option parameters including interest rate, volatility, etc.
        """
        self.model = model
        self.option = option

    def calculate(self, u: np.ndarray) -> np.ndarray:
        """
        Calculate the characteristic function for the given model.

        Parameters
        ----------
        u : np.ndarray
            Input array for the characteristic function.

        Returns
        -------
        np.ndarray
            The characteristic function values.
        """
        return self._vg_characteristic_function(u)

    def _vg_characteristic_function(self, u: np.ndarray) -> np.ndarray:
        """
        Calculate the characteristic function for the GBM model.

        Parameters
        ----------
        u : np.ndarray
            Input array for the characteristic function.

        Returns
        -------
        np.ndarray
            The characteristic function values.
        """
        mu = self.model.mu
        m = self.model.m
        delta = self.model.delta
        kappa = self.model.kappa
        r = self.option.r
        T = self.option.T
        emm = self.option.emm
        psi = self.option.psi

        if emm == "mean-correcting":
            b = r + np.log(1 - m*kappa - 0.5*kappa*delta**2)/kappa
            char_func = np.exp(
                T
                * (
                    1j * u * b
                    - np.log(1 - m*kappa * 1j*u + 0.5 *kappa *delta**2 *u**2)/kappa
                    )
            )
        elif emm == "Esscher":
            raise FeatureNotImplementedError(emm)
        else:
            raise UnsupportedEMMTypeError(emm)
        return char_func