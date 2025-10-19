from quantmetrics.price_calculators.characteristic_function_factory import get_characteristic_function

class CharacteristicFunction:
    """
    Wrapper class that automatically selects the correct characteristic function.

    Args:
        model: A LevyModel instance.
        option: An Option instance.
    """

    def __init__(self, model, option):
        self.model = model
        self.option = option
        self.cf = get_characteristic_function(model, option)

    def __call__(
        self,
        u,
        theta=None,
        exact=False,
        L=1e-12,
        M=1.0,
        N_center=150,
        N_tails=100,
        EXP_CLIP=700,
        search_bounds=(-50, 50),
        xtol=1e-8,
        rtol=1e-8,
        maxiter=500,
        sanity_theta=1.0,
        M_int=100,
        N_int=10_000,
        chunk_u=None
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
        return self.cf(
            u=u,
            theta=theta,
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
            sanity_theta=sanity_theta,
            M_int=M_int,
            N_int=N_int,
            chunk_u=chunk_u
        )
