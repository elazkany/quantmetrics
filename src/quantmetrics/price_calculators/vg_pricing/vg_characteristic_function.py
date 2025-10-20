
import numpy as np
from typing import TYPE_CHECKING

from quantmetrics.utils.exceptions import FeatureNotImplementedError, UnsupportedEMMTypeError



from quantmetrics.utils.integration import numeric_I_trap_array


if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option

class VGCharacteristicFunction:
    def __init__(self, model: "LevyModel", option: "Option"):
        """
        Initialize the VGCharacteristicFunction with a model and an option.

        Parameters
        ----------
        model : LevyModel
            The Levy model used for calculating the characteristic function.
        option : Option
            The option parameters including interest rate, volatility, etc.
        """
        self.model = model
        self.option = option

    def __call__(
            self,
            u: np.ndarray,
            exact = True,
            theta = None,
            L=1e-12,
            M=1.0,
            N_center=150,
            N_tails=100,
            EXP_CLIP=700,
            search_bounds = (-50, 50),
            xtol=1e-8,
            rtol=1e-8,
            maxiter=500,
            M_int = 100,
            N_int = 10_000,
            sanity_theta: float = 1.0,
            chunk_u = None,
            ) -> np.ndarray:
        """
        Calculate the characteristic function for the given model.

        Parameters
        ----------
        u : np.ndarray
            Input array for the characteristic function.

        Returns
        -------
        np.ndarray
            The characteristic function values.
        """
        return self._vg_characteristic_function(
            u=u,
            exact=exact,
            theta=theta,
            L=L,
            M=M,
            N_center=N_center,
            N_tails=N_tails,
            EXP_CLIP=EXP_CLIP,
            search_bounds=search_bounds,
            xtol=xtol,
            rtol=rtol,
            maxiter=maxiter,
            M_int=M_int,
            N_int=N_int,
            sanity_theta=sanity_theta,
            chunk_u = chunk_u,
        )

    def _vg_characteristic_function(
            self,
            u: np.ndarray,
            exact = False,
            theta = None,
            L=1e-12,
            M=1.0,
            N_center=150,
            N_tails=100,
            EXP_CLIP=700,
            search_bounds = (-50, 50),
            xtol=1e-8,
            rtol=1e-8,
            maxiter=500,
            sanity_theta: float = 1.0,
            M_int = 100,
            N_int = 10_000,
            chunk_u = None,
    ) -> np.ndarray:
        """
        Calculate the characteristic function for the GBM model.

        Parameters
        ----------
        u : np.ndarray
            Input array for the characteristic function.

        Returns
        -------
        np.ndarray
            The characteristic function values.
        """
        mu = self.model._mu
        m = self.model._m
        delta = self.model._delta
        kappa = self.model._kappa
        r = self.option.r
        T = self.option.T
        emm = self.option.emm
        psi = self.option.psi

        half = 0.5
        delta2 = delta * delta

        if emm == "mean-correcting":
            b = r + np.log(1 - m*kappa - half*kappa*delta2)/kappa
            return np.exp(
                T
                * (
                    1j * u * b
                    - np.log(1 - m*kappa * 1j*u + 0.5 *kappa *delta**2 *u**2)/kappa
                    )
            )
        elif emm == "Esscher":
            if theta is None:
                from quantmetrics.risk_neutral.market_price_of_risk import MarketPriceOfRisk
                theta = MarketPriceOfRisk(self.model, self.option).solve(
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
            b = mu + np.log(1 - m * kappa - half * delta2 * kappa) / kappa
            if psi == 0 and exact:
                def log_arg(theta):
                    return 1.0 - theta * m * kappa - half * delta2 * theta * theta * kappa

                A = log_arg(theta)
                B = log_arg(theta + 1j * u)

                return np.exp(
                    T
                    * (
                        1j * u * b
                        + (np.log(A) - np.log(B)) / kappa
                    )
                )

            elif not exact:
                I = numeric_I_trap_array(
                    u=u,
                    theta=theta,
                    psi=psi,
                    levy_density=self.model.levy_density,
                    M=M_int,
                    N=N_int,
                    EXP_CLIP=EXP_CLIP,
                    chunk_u = chunk_u,
                )
                return np.exp(T * (
                    1j * u * b + I
                    ))
            else:
                raise FeatureNotImplementedError("Exact integration for psi ≠ 0 is not implemented for VG.")
        else:
            raise UnsupportedEMMTypeError(emm)