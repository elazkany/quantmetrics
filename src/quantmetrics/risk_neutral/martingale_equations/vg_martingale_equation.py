from quantmetrics.utils.integration import integrate_split
import numpy as np


from quantmetrics.risk_neutral.martingale_equation_base import MartingaleEquation

from quantmetrics.option_pricing import Option
from quantmetrics.levy_models import VG

class VGMartingaleEquation(MartingaleEquation):
    """
    Martingale equation for the Variance Gamma (VG) model under Esscher EMM.

    Supports both exact and numerical integration approaches, and handles
    psi = 0 and psi ≠ 0 cases.
    """

    def __init__(self, model: VG, option: Option):
        if not isinstance(model, VG):
            raise TypeError("MartingaleEquationVG requires a VarianceGamma model.")
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
        m = self.model._m
        delta = self.model._delta
        kappa = self.model._kappa
        

        half = 0.5
        delta2 = delta * delta
        b = mu + np.log(1 - m * kappa - half * delta2 * kappa) / kappa

        if emm == 'mean-correcting':
                raise NotImplementedError(f"{emm} EMM is not implemented for VG")

        if emm == "Esscher":
            if psi == 0 and exact:
                def log_arg(theta):
                    return 1.0 - theta * m * kappa - half * delta2 * theta * theta * kappa

                A = log_arg(theta)
                B = log_arg(theta + 1)

                # guard against invalid domain: return a large sign
                if A <= 0.0 or B <= 0.0:
                    # return something with same sign as (b - r) to steer root solver away
                    # This helps bracketing solvers (like Brent’s method) detect that the root is not in this region
                    return np.sign(b - r) * 1e6

                return b - r + (np.log(A) - np.log(B)) / kappa

            elif not exact:
                def integrand(x):
                    exp1 = x
                    exp2 = theta * x + psi * x * x
                    y1 = np.exp(np.clip(exp1, -EXP_CLIP, EXP_CLIP))
                    y2 = np.exp(np.clip(exp2, -EXP_CLIP, EXP_CLIP))
                    return (y1 - 1.0) * y2 * self.model.levy_density(x)

                I = integrate_split(integrand, L=L, M=M, N_center=N_center, N_tails=N_tails)
                return np.inf if not np.isfinite(I) else b - r + I
            
            else:
                raise NotImplementedError("Exact integration for psi ≠ 0 is not implemented for VG.")

        else:
            raise NotImplementedError(f"EMM '{emm}' not supported for VG.")