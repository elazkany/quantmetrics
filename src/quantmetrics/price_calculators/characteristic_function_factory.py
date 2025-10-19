from quantmetrics.levy_models import GBM, CJD, LJD, VG
from quantmetrics.option_pricing import Option
from quantmetrics.price_calculators.vg_pricing.vg_characteristic_function import VGCharacteristicFunction
from quantmetrics.price_calculators.ljd_pricing.ljd_characteristic_function import LJDCharacteristicFunction
from quantmetrics.price_calculators.cjd_pricing.cjd_characteristic_function import CJDCharacteristicFunction
from quantmetrics.price_calculators.dejd_pricing.dejd_characteristic_function import DEJDCharacteristicFunction
from quantmetrics.price_calculators.gbm_pricing.gbm_characteristic_function import GBMCharacteristicFunction

from quantmetrics.risk_neutral.martingale_equation_base import MartingaleEquation

def get_characteristic_function(model, option: Option) -> MartingaleEquation:
    """
    Factory that returns the appropriate Characteristic function instance
    based on the model and option.

    Args:
        model: A LevyModel instance (e.g., VarianceGamma, LognormalJumpDiffusion)
        option: Option instance with EMM and psi

    Returns:
        MartingaleEquation: A model-specific martingale equation instance
    """
    if isinstance(model, VG):
        return VGCharacteristicFunction(model, option)
    elif isinstance(model, LJD):
        return LJDCharacteristicFunction(model, option)
    elif isinstance(model, CJD):
        return CJDCharacteristicFunction(model, option)
    elif isinstance(model, GBM):
        return GBMCharacteristicFunction(model, option)
    else:
        raise ValueError(f"No characteristic function available for model type: {type(model).__name__}")