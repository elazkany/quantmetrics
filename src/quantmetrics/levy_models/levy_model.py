from abc import ABC, abstractmethod
import logging

class LevyModel(ABC):
    def __init__(self, params: dict):
        """
        Initialize the LevyModel with parameters.

        Parameters:
        params (dict): Dictionary of model parameters.
        """
        self.params = params
        self.validate_params()
        self.setup_logging()

    def validate_params(self):
        """
        Validate the model parameters.
        """
        for key, value in self.params.items():
            if value is None:
                raise ValueError(f"Parameter {key} cannot be None")

    def setup_logging(self):
        """
        Set up logging for the model.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"{self.__class__.__name__} initialized with parameters: {self.params}")

    @abstractmethod
    
    def pdf(self):
        """
        Abstract method for the probability density function.
        """
        pass
