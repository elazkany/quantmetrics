# levy_models/geometric_brownian_motion.py
from .levy_model import LevyModel
import numpy as np
from scipy.optimize import minimize, brute
import scipy.stats as st

class GeometricBrownianMotion(LevyModel):
    def __init__(
        self,
        mu: float = 0.025,
        sigma: float = 0.02320006,
        S0: float=60,
        r: float = 0.08,
        q: float = 0,
        K: float = 65,
        t: float = 0,
        T: float = 0.25,
        payoff: str = "c",
        data: np.ndarray = None,
    ):
        """
        Geometric Brownian motion model

        Parameters
        ----------
        mu : float
            Expected return (drift)
        sigma : float
            Volatility (annualized)

            Divide by the square root of the number of days in a year (ex. 360) to convert to daily
        S0 : float
            Initial stock price
        r : float
            Interest rate (annualized)

            Divide by the number of days in a year (ex. 360) to convert to daily
        q : float
            Continuous dividend yield (annualized)

            Divide by the number of days in a year (ex. 360) to convert to daily
        t : float
            Current time
        T : float or int
            Time to maturity in years

            Multiply by the number of days in a year (ex. 360) to convert to daily
        payoff : str
            "c" for call option and "p" for put option
        data : np.array
            Data for fitting the model
        """

        params = {
            'S0': S0,
            'mu': mu,
            'sigma': sigma,
            'r': r,
            'q': q,
            'K': K,
            't': t,
            'T': T,
            'payoff': payoff,
        }
        super().__init__(params)
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.r = r
        self.q = q
        self.K = K
        self.t = t
        self.T = T
        self.payoff = payoff
        self.data = data

    def simulate_paths(
            self,
            numTimeSteps: int = 200,
            numPaths: int = 1000,
            seed: int = 1) -> dict:
        """
        Simulate Geometric Brownian Motion paths

        Parameters
        ----------
        numTimeSteps: int
            Number of time steps
        numPaths: int
            Number of simulated paths

        Returns
        -------
        paths: dict
            Dictionary with keys "time" and "S" of the simulated paths
        """
        np.random.seed(seed)
        dt = self.T / float(numTimeSteps)
        Z = np.random.standard_normal(
            size=(numTimeSteps, numPaths)
        )  # generate standard normal random variable of size.... [t,omega]
        # Standard Brownian motion
        W = np.zeros((numTimeSteps + 1, numPaths))
        # Return process under Q (risk-neutral)
        X = np.zeros(W.shape)
        X[0, :] = np.log(self.S0)
        time = np.zeros(W.shape[0])

        for i in range(0, numTimeSteps):
            # Making sure that samples from the normal distribution have mean 0 and variance 1
            if numPaths > 1:
                Z[i, :] = (Z[i, :] - np.mean(Z[i, :])) / np.std(Z[i, :])
            W[i + 1, :] = W[i, :] + np.power(dt, 0.5) * Z[i, :]
            X[i + 1, :] = (
                X[i, :]
                + (self.r - self.q - 0.5 * self.sigma**2) * dt
                + self.sigma * (W[i + 1, :] - W[i, :])
            )
            time[i + 1] = time[i] + dt

        # Compute exponent of ABM
        S = np.exp(X)
        paths = {"time": time, "S": S}
        return paths

    def pdf(
            self,
            est_params: np.ndarray
            ):
        mu, sigma = est_params
        if sigma <= 0.0:
            return 500.0
        else:
            drift = mu - 0.5 * sigma**2
            return st.norm.pdf(self.data, loc=drift, scale=sigma)

    def fit(self):
        def MLE(params):
            return -np.sum(
                np.log(
                    self.pdf(
                        params=params,
                    )
                )
            )

        params = brute(
            MLE,
            (
                (-1, 1, 0.5),  # mu
                (0.05, 2, 0.5),  # sigma
            ),
            finish=None,
        )

        result = minimize(MLE, params, method="Nelder-Mead")
        # mu, sigma = result.x
        return result