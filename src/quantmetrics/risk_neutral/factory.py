from quantmetrics.risk_neutral.martingale_equations.vg_martingale_equation import VGMartingaleEquation
from quantmetrics.risk_neutral.martingale_equations.ljd_martingale_equation import LJDMartingaleEquation
from quantmetrics.risk_neutral.martingale_equations.cjd_martingale_equation import CJDMartingaleEquation
from quantmetrics.risk_neutral.martingale_equations.gbm_martingale_equation import GBMMartingaleEquation

from quantmetrics.option_pricing import Option
from quantmetrics.levy_models import GBM, CJD, LJD, VG
from quantmetrics.risk_neutral.martingale_equation_base import MartingaleEquation

def get_martingale_equation(model, option: Option) -> MartingaleEquation:
    """
    Factory that returns the appropriate MartingaleEquation instance
    based on the model and option.

    Args:
        model: A LevyModel instance (e.g., VarianceGamma, LognormalJumpDiffusion)
        option: Option instance with EMM and psi

    Returns:
        MartingaleEquation: A model-specific martingale equation instance
    """
    if isinstance(model, VG):
        return VGMartingaleEquation(model, option)
    elif isinstance(model, LJD):
        return LJDMartingaleEquation(model, option)
    elif isinstance(model, CJD):
        return CJDMartingaleEquation(model, option)
    elif isinstance(model, GBM):
        return GBMMartingaleEquation(model, option)
    else:
        raise ValueError(f"No martingale equation available for model type: {type(model).__name__}")