from quantmetrics.utils.integration import integrate_split
from quantmetrics.risk_neutral.martingale_equations.martingale_equation_base import MartingaleEquation
from quantmetrics.levy_models import LJD
from quantmetrics.option_pricing import Option
import numpy as np

class LJDMartingaleEquation(MartingaleEquation):
    """
    Martingale equation for the lognormal jump-diffusion (LJD) model under Esscher EMM.

    Supports both exact and numerical integration.
    """

    def __init__(self, model: LJD, option: Option):
        if not isinstance(model, LJD):
            raise TypeError("LJDMartingaleEquation requires a LogNormal Jump-Diffusion model.")
        super().__init__(model, option)

    def evaluate(
            self,
            theta: float,
            exact: bool = False,
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
        muJ = self.model._muJ
        sigmaJ = self.model._sigmaJ

        sigma2 = sigma * sigma

        if emm == 'mean-correcting':
            return mu - r + theta * sigma2

        if emm == "Esscher":
            if exact:
                sigmaJ2 = sigmaJ * sigmaJ
                muJ2 = muJ * muJ
                g_psi = 1 -2 * sigmaJ2 * psi
                f = lambda x: np.exp(
                    (muJ * x + 0.5 * sigmaJ2 * x*x + psi * muJ2) / g_psi
                ) / np.sqrt(g_psi)

                exp1 = np.exp(muJ + sigmaJ2/2)

                return mu - r + theta * sigma2 + lambda_ * ( f(theta + 1) - f(theta) - (exp1-1))

            else:
                def integrand(x):
                    exp1 = x
                    exp2 = theta * x + psi * x * x
                    y1 = np.exp(np.clip(exp1, -EXP_CLIP, EXP_CLIP))
                    y2 = np.exp(np.clip(exp2, -EXP_CLIP, EXP_CLIP))
                    return (y1 - 1.0) * (y2 - 1.0) * self.model.levy_density(x)

                I = integrate_split(integrand, L=L, M=M, N_center=N_center, N_tails=N_tails)
                return np.inf if not np.isfinite(I) else mu - r + theta * sigma2 + lambda_ * I

        else:
            raise NotImplementedError(f"EMM '{emm}' not supported for LJD.")