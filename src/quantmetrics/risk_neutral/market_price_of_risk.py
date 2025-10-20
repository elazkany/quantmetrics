from quantmetrics.risk_neutral.factory import get_martingale_equation
from quantmetrics.risk_neutral.market_price_of_risk_solver import MarketPriceOfRiskSolver

class MarketPriceOfRisk:
    """
    Wrapper class that automatically selects the correct martingale equation
    and solves for the market price of risk (theta).

    Args:
        model: A LevyModel instance.
        option: An Option instance.
    """

    def __init__(self, model, option):
        self.equation = get_martingale_equation(model, option)
        self.solver = MarketPriceOfRiskSolver(self.equation)

    def solve(
        self,
        exact: bool =True,
        L=1e-12,
        M=1.0,
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
        Solves the martingale equation for theta.

        Args:
            exact (bool): Use closed-form expression if available.
            search_bounds (tuple): Bracketing interval for root-finding.
            xtol (float): Absolute tolerance.
            rtol (float): Relative tolerance.
            maxiter (int): Max iterations.
            sanity_theta (float): Theta value to test for finite evaluation.

        Returns:
            float: Market price of risk (theta).
        """
        return self.solver.solve(
            exact=exact,
            L=L,
            M=M,
            N_center=N_center,
            N_tails=N_tails,
            EXP_CLIP=EXP_CLIP,
            search_bounds=search_bounds,
            xtol=xtol,
            rtol=rtol,
            maxiter=maxiter,
            sanity_theta=sanity_theta
        )