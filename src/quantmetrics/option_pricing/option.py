#option_pricing/option.py

class Option:
    def __init__(
            self,
            r: float = 0.05/365,
            q: float = 0,
            K: float = 60,
            T: float = 63,
            payoff: str = "c",
            emm : str = "Black-Scholes",
            psi : float = 0,
            ):
        """
        Initialize option data

        Parameters
        ----------
        r : float
            Interest rate (annualized)

            Divide by the number of days in a year (ex. 360) to convert to daily
        q : float
            Continuous dividend yield (annualized)

            Divide by the number of days in a year (ex. 360) to convert to daily
        K : float
            Strike price of the option
        T : float or int
            Time to maturity in years

            Multiply by the number of days in a year (ex. 360) to convert to daily
        payoff : str
            "c" for call option and "p" for put option
        emm : str
            The equivalent martingale measure used for pricing the option. 
            Options include:
            - "Black-Scholes": The standard measure used in the Black-Scholes model, also the measure applied to Merton's jump-diffusion model.
            - "Classical Esscher": A measure more general than the "Black-Scholes" measure which includes, for example, pricing jump risk.
            - "Second-order Esscher": A measure more general than the "Classical Esscher", see References for more details.

        psi: float
            The free parameter of the second-order Esscher EMM. Default is 0 which corresponds to the classical Esscher EMM. See References for details.

        References
        ----------
        Choulli, T., Elazkany, E., & Vanmaele, M. (2024). The second-order Esscher martingale densities for continuous-time market models. arXiv preprint arXiv:2407.03960.
        """

        
        self.r = r
        self.q = q
        self.K = K
        self.T = T
        self.payoff = payoff
        self.emm = emm
        self.psi = psi