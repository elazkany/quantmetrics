#src\quantmetrics\price_calculators\ljd_pricing\ljd_paths_Q.py

import numpy as np
from typing import TYPE_CHECKING, Dict

from quantmetrics.utils.exceptions import UnsupportedEMMTypeError
from quantmetrics.risk_calculators.martingale_equation import RiskPremium

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option

class LJDSimulatePathsQ:
    """
    Implements the paths simulation for a lognormal jump-diffusion (LJD) model under the risk-neutral measure.

    Parameters
    ----------
    model : LevyModel
        A LevyModel object specifying the underlying asset's model and its parameters.
    option : Option
        An Option object specifying the option parameters: interest rate, strike price, time to maturity, dividend yield and the equivalent martingale measure.
    """
    def __init__(self, model: "LevyModel", option: "Option"):
        self.model = model
        self.option = option

    def simulate(self, num_timesteps: int, num_paths: int, seed: int) -> Dict[str, np.ndarray]:
        """
        Generate paths for the lognormal jump-diffusion model.

        Parameters
        ----------
        num_timesteps : int
            Number of time steps.
        num_paths : int
            Number of simulated paths.
        seed : int
            Seed for random number generator.

        Returns
        -------
        dict
            A dictionary containing:

            - `time_steps` (np.ndarray): The simulated time steps.
            - `S_exact` (np.ndarray): The simulated LJD paths using the exact solution.
            - `S_euler` (np.ndarray): The simulated LJD paths using the Euler-Maruyama discretization.

        
        Notes
        -------

        The Euler-Maruyama discretization for the :math:`i^{th}` timestep and :math:`j^{th}` path, reads:

        .. math::

            s_{i+1, j} \\approx s_{i,j} + (r-q - \\lambda^\\mathbb{Q} \\kappa^\\mathbb{Q}) s_{i,j} \\Delta t + \\sigma s_{i,j} (W_{i+1, j} - W_{i,j}) + J s_{i,j} (N_{i+1,j} -N_{i,j}),

        with :math:`\\Delta t = t_{i+1} - t_i`, for any :math:`i=1,2,\\cdots , m, \\ s_0 = S(t_0)=S_0`, :math:`j = 1,2,\\cdots, N`, :math:`W_{i+1, j} - W_{i,j} \\sim \\mathcal{N}(0, \Delta t)`, :math:`J \\sim \\mathcal{N}(\\mu_J, \\sigma_J^2)` and :math:`N_{i+1,j} -N_{i,j} \sim \\mathcal{Poi}(\\lambda^\\mathbb{Q} \\Delta t)`. 

        The LJD process has as exact solution in the time interval :math:`[t_i, t_{i+1}]`,

        .. math::

            S(t_{i+1})=S(t_i)\\exp\\left\{(r-q-\\frac{\\sigma^2}{2}- \\lambda^\\mathbb{Q} \\kappa^\\mathbb{Q})\\Delta t + \\sigma [W(t_{i+1}) - W(t_i)] + J [N(t_{i+1}) - N(t_i)] \\right\}
            
        where

            - :math:`S_0` is the underlying price.
            - :math:`r` is the risk-free interest rate.
            - :math:`q` is the dividend yield. 
            - :math:`\\sigma` is the volatility.
            - :math:`\\mu_J` is the mean of the jump sizes.
            - :math:`\\sigma_J` is the standard deviation of the jump sizes.
            - :math:`\\lambda` is the jump intensity rate.

        The parameters :math:`\\lambda^\\mathbb{Q}` and :math:`\\kappa^\\mathbb{Q}` depend on the choice of the equivalent martingale measure.

        If ``emm = "mean-correcting"`` then

        .. math::

            \\lambda^\\mathbb{Q} = \\lambda \\quad \\text{and} \\quad \\kappa^\\mathbb{Q} = \\kappa = \\exp \\left(\\mu_J + \\frac{\\sigma_J^2}{2} \\right) - 1

        If ``emm = "Esscher"`` then

        .. math::

            \\lambda^\\mathbb{Q} = \\lambda f(\\theta) \\quad \\text{and} \\quad \\kappa^\\mathbb{Q} = \\frac{f(\\theta +1) }{f(\\theta)} - 1 = \\left[ \\left(\\kappa +1 \\right)e^{\\theta \\sigma_J^2}\\right]^{\\frac{1}{g(\\psi)}} -1

        where 
        
        .. math::

            f(y) = \\frac{1}{\\sqrt{g(\\psi)}} \\exp \\left[\\frac{1}{g(\\psi)} \\left(\\mu_J y + \\frac{\\sigma_J^2}{2} y^2 + \\psi \\mu_J^2  \\right)  \\right],\\quad g(\\psi) = 1 - 2\\psi \\sigma_J^2

        with

        .. math::

            \\psi < \\frac{1}{2\\sigma_J^2}
        
        The first-order Esscher parameter :math:`\\theta` is the risk premium (market price of risk) and which is the unique solution to the martingale equation for each :math:`\\psi` which is the second-order Esscher parameter. See the documentation of the ``RiskPremium`` class for the martingale equation and refer to [1]_ for more details.
        
       
        Examples
        --------
        >>> from quantmetrics.levy_models import LJD
        >>> from quantmetrics.option_pricing import Option
        >>> from quantmetrics.price_calculators.ljd_pricing.ljd_paths_Q import LJDSimulatePathsQ
        >>> ljd = LJD() # S0=50, sigma=0.2, lambda_=1, muJ=-0.1, sigmaJ=0.1
        >>> option = Option(K=np.array([20,50,80]), T = 20/252) # r=0.05, q=0.02
        >>> paths = LJDSimulatePathsQ(ljd, option).simulate(num_timesteps=200, num_paths=10000,seed=42)
        >>> payoff = np.maximum(paths["S_exact"][:,-1].reshape(-1,1) - option.K, 0)
        >>> option_price = np.mean(np.exp(-option.r*option.T) * payoff, axis=0)
        >>> option_price
        array([30.01448984,  1.32307021,  0.        ])

        References
        ----------

        .. [1] Choulli, T., Elazkany, E., & Vanmaele, M. (2024). Applications of the Second-Order Esscher Pricing in Risk Management. arXiv preprint arXiv:2410.21649.
        
        .. [2] Matsuda, K. (2004). Introduction to option pricing with Fourier transform: Option pricing with exponential Lévy models. Department of Economics The Graduate Center, The City University of New York, 1-241.

        .. [3] Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. Journal of financial economics, 3(1-2), 125-144.

        .. [4] Oosterlee, C. W., & Grzelak, L. A. (2019). Mathematical modeling and computation in finance: with exercises and Python and MATLAB computer codes. World Scientific.
        """
        return self._ljd_simulate_paths_Q(num_timesteps, num_paths, seed)

    def _ljd_simulate_paths_Q(self, num_timesteps: int, num_paths: int, seed: int) -> Dict[str, np.ndarray]:
        
        np.random.seed(seed)

        S0 = self.model.S0
        sigma = self.model.sigma
        lambda_ = self.model.lambda_
        muJ = self.model.muJ
        sigmaJ = self.model.sigmaJ
        r = self.option.r
        T = self.option.T
        q = self.option.q
        emm = self.option.emm
        psi = self.option.psi

        np.random.seed(seed)

        dt = T / float(num_timesteps)
        """
        Z = np.random.standard_normal(
            size=(num_paths,num_timesteps)
        )

        W = np.zeros((num_paths, num_timesteps + 1 ))
        
        S_exact = np.zeros(W.shape)
        S_exact[:, 0] = S0
        S_euler = np.zeros(W.shape)
        S_euler[:, 0] = S0

        time = np.zeros(W.shape[0])

        if emm == "mean-correcting":
            lambda_Q = lambda_
            kappa = np.exp(muJ + 0.5 * sigmaJ**2) - 1
        else:
            theta = RiskPremium(self.model, self.option).calculate()

            g_psi = 1 - 2 * psi * sigmaJ**2

            f = lambda x: np.exp(
                (muJ * x + 0.5 * sigmaJ**2 * x**2 + psi * muJ**2) / g_psi
            ) / (g_psi**0.5)

            lambda_Q = lambda_ * f(theta) 

            kappa = f(theta + 1) / f(theta) - 1

        ZPois = np.random.poisson(lambda_Q * dt, [num_paths, num_timesteps])

        J = np.random.normal(muJ, sigmaJ, [num_paths, num_timesteps])

        for i in range(0, num_timesteps):
            # Making sure that samples from a normal have mean 0 and variance 1
            if num_paths > 1:
                Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
                # Making sure that samples from a normal have mean 0 and variance 1
            W[:,i + 1] = W[:,i] + np.power(dt, 0.5) * Z[:,i]

            S_exact[:,i + 1] = S_exact[:,i] * np.exp( (r - q -0.5 *sigma*sigma - lambda_Q * kappa)*dt + sigma * (W[:,i + 1] - W[:,i])+ J[:,i] * ZPois[:, i])
                    
            S_euler[:,i + 1] = S_euler[:,i] + (r-q - lambda_Q * kappa) * S_euler[:,i] * dt + sigma * S_euler[:,i] * (W[:,i + 1] - W[:,i]) +(np.exp(J[:,i])-1)* S_euler[:,i] * ZPois[:,i]

            time[i + 1] = time[i] + dt

        paths = {"time": time, "S_exact": S_exact, "S_euler": S_euler
        """

        # Generate normally distributed random variables with shape (num_paths, num_timesteps)
        Z = np.random.standard_normal(size=(num_paths, num_timesteps))
        
        # Normalize each column so that each time-step's ensemble has mean 0 and variance 1.
        Z = (Z - np.mean(Z, axis=0)) / np.std(Z, axis=0)
        
        # Compute the increments of Brownian motion: dW = sqrt(dt) * Z.
        dW = np.sqrt(dt) * Z

        # Solve the martingale equation for the selected EMM
        if emm == "mean-correcting":
            lambda_Q = lambda_
            kappa = np.exp(muJ + 0.5 * sigmaJ**2) - 1
        elif emm == "Esscher":
            theta = RiskPremium(self.model, self.option).calculate()

            g_psi = 1 - 2 * psi * sigmaJ**2

            f = lambda x: np.exp(
                (muJ * x + 0.5 * sigmaJ**2 * x**2 + psi * muJ**2) / g_psi
            ) / (g_psi**0.5)

            lambda_Q = lambda_ * f(theta) 

            # lambda_Q * kappa * dt is the compensator of the compound Poisson process
            kappa = f(theta + 1) / f(theta) - 1

        else:
            raise UnsupportedEMMTypeError(emm)
        

        # Generate Poisson distributed random variables with intensity lambda_Q * dt
        ZPois = np.random.poisson(lambda_Q * dt, [num_paths, num_timesteps])

        # Generate normally distributed jump sizes
        J = np.random.normal(muJ, sigmaJ, [num_paths, num_timesteps])

        # For the exact solution, the multiplicative increment for each time step is:
        # exp((r - q - 0.5*sigma^2)*dt + sigma*dW)
        incr_exact = np.exp((r - q - 0.5 * sigma**2 - lambda_Q * kappa) * dt + sigma * dW + J * ZPois)
        # The exact solution is obtained via cumulative product:
        S_exact = S0 * np.concatenate((np.ones((num_paths, 1)), np.cumprod(incr_exact, axis=1)), axis=1)
    
        # For the Euler approximation, the update is:
        # S_euler[i+1] = S_euler[i] * [1 + (r - q) * dt + sigma*dW]
        incr_euler = 1 + (r - q - lambda_Q * kappa) * dt + sigma * dW + (np.exp(J)-1)* ZPois
        S_euler = S0 * np.concatenate((np.ones((num_paths, 1)), np.cumprod(incr_euler, axis=1)), axis=1)
    
        # Create a time array from 0 to T with (num_timesteps+1) points
        time = np.linspace(0, T, num_timesteps + 1)
    
        paths = {"time": time, "S_exact": S_exact, "S_euler": S_euler}

        return paths
