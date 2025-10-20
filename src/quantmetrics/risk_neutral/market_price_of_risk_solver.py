import numpy as np
import scipy.optimize
from quantmetrics.risk_neutral.martingale_equation_base import MartingaleEquation

class MarketPriceOfRiskSolver:
    """
    Solver for the market price of risk (theta) using a martingale equation.

    This class performs root-finding to solve the martingale equation defined
    by a model-specific MartingaleEquation instance.

    Args:
        equation (MartingaleEquation): Instance of a model-specific martingale equation.
    """

    def __init__(self, equation: MartingaleEquation):
        self.equation = equation

    def solve(
        self,
        exact=True,
        L=1e-12,
        M=20.0,
        N_center=150,
        N_tails=100,
        EXP_CLIP=700,
        search_bounds: tuple = (-50.0, 50.0),
        xtol: float = 1e-8,
        rtol: float = 1e-8,
        maxiter: int = 500,
        sanity_theta: float = 1.0
    ) -> float:
        """
        Finds the value of theta that solves the martingale equation.

        Args:
            exact (bool): Use closed-form expression if available.
            search_bounds (tuple): Bracketing interval (lower, upper) for root-finding.
            xtol (float): Absolute tolerance for convergence.
            rtol (float): Relative tolerance for convergence.
            maxiter (int): Maximum number of iterations.
            sanity_theta (float): Theta value to test for finite evaluation.

        Returns:
            float: Root theta such that martingale equation = 0.
        """
        test_val = self.equation.evaluate(sanity_theta)
        if not np.isfinite(test_val):
            raise RuntimeError(
                f"Martingale equation not finite at theta={sanity_theta} (value={test_val}). "
                "Try adjusting model parameters or integration settings."
            )

        def wrapped(theta):
            return self.equation.evaluate(
                theta,
                exact=exact,
                L=L,
                M=M,
                N_center=N_center,
                N_tails=N_tails,
                EXP_CLIP=EXP_CLIP,
                )

        try:
            theta = scipy.optimize.brentq(
                wrapped,
                search_bounds[0],
                search_bounds[1],
                xtol=xtol,
                rtol=rtol,
                maxiter=maxiter
            )
        except ValueError:
            theta = np.inf

        return theta