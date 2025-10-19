from abc import ABC, abstractmethod
from quantmetrics.levy_models.levy_model_base import LevyModel
from quantmetrics.option_pricing import Option

class MartingaleEquation(ABC):
    """
    Abstract base class for martingale equations under different Lévy models.

    Subclasses must implement the `evaluate` method, which computes the value
    of the martingale equation for a given theta.

    Attributes:
        model (LevyModel): The underlying Lévy model.
        option (Option): The option or market context, including EMM type and psi.
    """

    def __init__(self, model: LevyModel, option: Option):
        self.model = model
        self.option = option

    @abstractmethod
    def evaluate(self, theta: float, exact: bool = False) -> float:
        """
        Evaluate the martingale equation at a given theta.

        Args:
            theta (float): The Esscher parameter or market price of risk.
            exact (bool): Whether to use a closed-form expression if available.

        Returns:
            float: Value of the martingale equation at theta.
        """
        pass