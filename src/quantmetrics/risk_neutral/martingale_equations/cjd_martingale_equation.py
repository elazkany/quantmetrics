import numpy as np
from quantmetrics.option_pricing import Option
from quantmetrics.levy_models import CJD
from quantmetrics.risk_neutral.martingale_equation_base import MartingaleEquation


class CJDMartingaleEquation(MartingaleEquation):
    """
    Martingale equation for the constant jump-diffusion (CJD) model under Esscher EMM.

    Supports both exact and numerical integration.
    """

    def __init__(self, model: CJD, option: Option):
        if not isinstance(model, CJD):
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
        emm = self.option.emm
        psi = self.option.psi
        mu = self.model._mu
        sigma = self.model._sigma
        lambda_ = self.model._lambda_
        gamma = self.model._gamma

        sigma2 = sigma * sigma

        gamma_tilde = np.exp(gamma) - 1

        if emm == 'mean-correcting':
            return mu - r + theta * sigma2

        elif emm == "Esscher":
            exp0 = theta * gamma + psi * gamma * gamma
            exp_clipped = np.exp(np.clip(exp0, -EXP_CLIP, EXP_CLIP))
            return mu - r + theta * sigma2 + lambda_ * gamma_tilde * (exp_clipped - 1)

        else:
            raise NotImplementedError(f"EMM '{emm}' not supported for CJD.")