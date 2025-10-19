import numpy as np
from scipy.fft import fft

from quantmetrics.levy_models import GBM, CJD, LJD, DEJD, VG
from quantmetrics.price_calculators.gbm_pricing.gbm_calculator import GBMCalculator
from quantmetrics.price_calculators.cjd_pricing.cjd_calculator import CJDCalculator
from quantmetrics.price_calculators.ljd_pricing.ljd_calculator import LJDCalculator
from quantmetrics.price_calculators.dejd_pricing.dejd_calculator import DEJDCalculator
from quantmetrics.price_calculators.vg_pricing.vg_calculator import VGCalculator

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option
    from quantmetrics.price_calculators import BaseCalculator

class OptionPricer:
    def __init__(self, model: "LevyModel", option: "Option"):
        self.model = model
        self.option = option
        self.calculator = self._get_calculator()

    def _get_calculator(self) -> "BaseCalculator":
        if isinstance(self.model, GBM):
            return GBMCalculator(self.model, self.option)
        
        elif isinstance(self.model, CJD):
            return CJDCalculator(self.model, self.option)
        
        elif isinstance(self.model, LJD):
            return LJDCalculator(self.model, self.option)
        
        elif isinstance(self.model, DEJD):
            return DEJDCalculator(self.model, self.option)
        
        elif isinstance(self.model, VG):
            return VGCalculator(self.model, self.option)

        else:
            raise ValueError(f"Unknown model type: {type(self.model).__name__}")
        
    """
        
        
        
        elif isinstance(self.model, NIG):
            return NIGCalculator(self.model, self.option)
        elif isinstance(self.model, HestonModel):
            return HestonCalculator(self.model, self.option)
        """
        

    def closed_form(self) -> Union[float, np.ndarray]:
        return self.calculator.calculate_closed_form()

    def fft(self,
        exact=False,
        L=1e-16, # Center bound of integral in M.E.
        M = 2.0, # Upper tail bound of integral in M.E.
        N_center = 150, # Integration samples in M.E.
        N_tails = 100, # Integration samples in M.E.
        EXP_CLIP = 700, # Avoid overflow in exponential
        search_bounds = (-100, 100), # For the theta root finder
        xtol = 1e-8, # For the theta root finder
        rtol = 1e-8, # For the theta root finder
        maxiter = 500, # For the theta root finder
        sanity_theta = 1.0, # For the theta root finder
        M_int = 1, # Integral truncation for CF integral
        N_int = 1_000, # Integral samples for CF integral
        g = 2, # Accuracy factor for FFT grid
        N = None, # Number of FFT points (default: g*2^10)
        eps = None, # FFT grid spacing (default: 1/(g*150))
        eta = None, # FFT frequency spacing (default: 2*pi/(N*eps))
        alpha_itm = 1.5, # Dampening factor for ITM options
        alpha_otm = 1.1, # Dampening factor for OTM options
    ):
        S0 = self.model.S0
        K = self.option.K
        r = self.option.r
        T = self.option.T
        q = self.option.q
        #print("Starting FFT pricer...")

        from quantmetrics.risk_neutral.market_price_of_risk import MarketPriceOfRisk
        mpr = MarketPriceOfRisk(self.model, self.option)
        theta = mpr.solve(
                exact=exact,
                L=L,
                M=M,
                N_center=N_center,
                N_tails=N_tails,
                EXP_CLIP=EXP_CLIP,
                search_bounds = search_bounds,
                xtol=xtol,
                rtol=rtol,
                maxiter=maxiter,
                sanity_theta=sanity_theta)

        if alpha_itm <= 1.0 or alpha_otm <= 1.0:
                raise ValueError("alpha must be > 1.0 to ensure integrability")

        if np.isscalar(K):
            K = np.array([K])
        else:
            K = np.asarray(K)

        prices = np.zeros(K.shape, dtype=float)

        if N is None:
            N = g * 2**12 # default number of FFT points
        if eps is None:
            eps = (g * 150.0) ** -1 # default log-strike grid spacing
        if eta is None:
            eta = 2 * np.pi / (N * eps) # default frequency spacing

        # Log-strike grid spacing
        k = np.log(K / S0 * np.exp(q*T)) 

        # FFT frequency grid
        v0 = eta * (np.arange(1, N + 1, 1) - 1) # v_j = eta * (j - 1) has shape (N,)

        # FFT log-strike grid offset
        b = 0.5 * N * eps - k # has shape same as k
        
        # Precompute Simpson-like weights
        delta = np.zeros(N)
        delta[0] = 1.0
        j = np.arange(1, N + 1)
        simpson_weights = (3 + (-1)**j - delta) / 3.0 # has shape (N,)
        eta_simpson = eta * simpson_weights

        # ITM
        omega_itm = v0 - (alpha_itm + 1) * 1j
        from quantmetrics.price_calculators.characteristic_function import CharacteristicFunction
        char_func_itm = self.calculator.calculate_characteristic_function(
                omega_itm,
                exact,
                theta,
                L,
                M,
                N_center,
                N_tails,
                EXP_CLIP,
                search_bounds,
                xtol,
                rtol,
                maxiter,
                M_int,
                N_int
                )
        denom_itm = alpha_itm * alpha_itm + alpha_itm - v0 * v0 + 1j * (2 * alpha_itm + 1) * v0
        Psi_itm = np.exp(-r * T) * char_func_itm / denom_itm
            
        # OTM
        v_minus = v0 - 1j * alpha_otm
        v_plus = v0 + 1j * alpha_otm
        omega_otm1 = v_minus - 1j
        omega_otm2 = v_plus - 1j

        char_func_otm1 = self.calculator.calculate_characteristic_function(
                omega_otm1,
                exact,
                theta,
                L,
                M,
                N_center,
                N_tails,
                EXP_CLIP,
                search_bounds,
                xtol,
                rtol,
                maxiter,
                M_int,
                N_int
                )
        
        char_func_otm2 = self.calculator.calculate_characteristic_function(
                omega_otm2,
                exact,
                theta,
                L,
                M,
                N_center,
                N_tails,
                EXP_CLIP,
                search_bounds,
                xtol,
                rtol,
                maxiter,
                M_int,
                N_int
                )
        
        denom_otm1 = v_minus** 2 - 1j * v_minus

        Psi_otm1 = np.exp(-r * T) * (
                    1 / (1 + 1j * v_minus)
                    - np.exp(r * T) / (1j * v_minus)
                    - char_func_otm1
                    / denom_otm1)
        
        denom_otm2 = v_plus** 2 - 1j * v_plus
                
        Psi_otm2 = np.exp(-r * T) * (
                    1 / (1 + 1j * v_plus)
                    - np.exp(r * T) / (1j * v_plus)
                    - char_func_otm2 / denom_otm2
                )
        
        phase_base = v0
        K_arr = np.atleast_1d(np.asarray(K, dtype=float))
        scalar_input = (K_arr.size == 1)
        prices = np.zeros_like(K_arr, dtype=float)

        for idx, Kval in enumerate(K_arr):
            k = np.log(Kval / S0 * np.exp(q * T))  # original k
            # original set g=1 => N=4096 used above
            b = 0.5 * N * eps - k         # per-strike b as original
            pos = int((k + b) / eps)     # original integer pos extraction
        
            # decide ITM vs OTM exactly as original
            if S0 * np.exp(-q * T) >= 0.95 * Kval:
                # ITM case: use modchar_itm
                FFTFunc = np.exp(1j * b * phase_base) * Psi_itm * eta_simpson
                payoff = np.real(fft(FFTFunc))
                # original computes CallValueM = exp(-alpha_itm*k)/pi * payoff
                CallValueM = np.exp(-alpha_itm * k) / np.pi * payoff
            else:
                # OTM case: use modchar_otm1 and modchar_otm2
                FFTFunc = np.exp(1j * b * phase_base) * (Psi_otm1 - Psi_otm2) * 0.5 * eta_simpson
                payoff = np.real(fft(FFTFunc))
                # handle small-k safely (ks == k can be zero)
                denom_sinh = np.sinh(alpha_otm * k)
            
                if np.abs(k) < 1e-12:
                    denom_sinh_safe = alpha_otm * k
                else:
                    denom_sinh_safe = denom_sinh
                # guard remaining exact zero
                if denom_sinh_safe == 0.0:
                    denom_sinh_safe = alpha_otm * 1e-300
            
                CallValueM = payoff / (denom_sinh * np.pi)

            # original picks the pos element
            # ensure pos is in-bounds (original code assumed pos valid)
            pos_clamped = max(0, min(pos, N - 1))
            CallValue = CallValueM[pos_clamped] * S0 * np.exp(-q * T)

            prices[idx] = 0.0 if CallValue <= 0.0 else CallValue
    
        return [float(prices[0]) if scalar_input else prices, theta]
        
    def monte_carlo(self, num_timesteps: int = 200, num_paths: int = 10000, seed: int = 42, exact_solution = True) -> Union[float, np.ndarray]:
        """
        Calculate the Monte Carlo price for the given option.

        Parameters
        ----------
        num_timesteps : int, optional
            Number of time steps (default is 200).
        num_paths : int, optional
            Number of simulated paths (default is 10000).
        seed : int, optional
            Seed for random number generator (default is 42).

        print_time : bool, optional
        Whether to print the elapsed time (default is False).

        Returns
        -------
        np.ndarray
            A two-dimensional array containing the estimated price of the option and the standard error.
        """
        r = self.option.r
        K = self.option.K
        T = self.option.T

        paths = self.calculator.simulate_paths_Q(num_timesteps, num_paths, seed)

        if exact_solution == True:
            S = paths["S_exact"]
        else:
            S = paths["S_euler"]
        
        payoff = np.maximum(S[:,-1].reshape(-1,1) - K, 0)

        mc_price = np.mean(np.exp(-r * T) * payoff, axis=0)

        # Calculate the sample variance
        sample_var = np.sum((payoff - mc_price) ** 2, axis=0) / (num_paths - 1)

        # Calculate the standard error
        standard_error = (sample_var / num_paths) ** 0.5

        return np.array([mc_price, standard_error])
