
from quantmetrics.option_pricing import Option
from quantmetrics.levy_models import GBM
from quantmetrics.risk_neutral.martingale_equation_base import MartingaleEquation


class GBMMartingaleEquation(MartingaleEquation):
    """
    Martingale equation for the geometric Brownian motion (GBM) model under Esscher EMM.
    """

    def __init__(self, model: GBM, option: Option):
        if not isinstance(model, GBM):
            raise TypeError("CJDMartingaleEquation requires a constant jump-diffusion model.")
        super().__init__(model, option)

    def evaluate(
            self,
            theta: float,
            exact: bool = True,
            EXP_CLIP = 700,
            L=1e-12,
            M=1.0,
            N_center=150,
            N_tails=100
            ) -> float:
        r = self.option.r
        
        mu = self.model._mu
        sigma = self.model._sigma
        sigma2 = sigma * sigma

        return mu - r + theta * sigma2