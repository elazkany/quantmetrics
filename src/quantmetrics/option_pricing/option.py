#option_pricing/option.py

class Option:
    def __init__(
            self,
            r: float = 0.05,
            q: float = 0.02,
            K: float = 50.0,
            T: float = 0.5,
            option_type : str = "e",
            payoff: str = "c",
            emm : str = "mean-correcting",
            psi : float = 0.0,
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
        option_type : str
            Option type, "e" for European options and "a" for American options.
        payoff : str
            "c" for call option and "p" for put option
        emm : str
            The equivalent martingale measure used for pricing the option. 
            Options include:
            - "mean-correcting": Mean-correcting measure which is the equivalent martingale measure used in the Black-Scholes model.
            - "Esscher": A measure more general than the "mean-correcting" measure which includes, for example, pricing jump risk.

        psi: float
            A free parameter of the second-order Esscher EMM. Default is 0 which corresponds to the classical Esscher EMM. See References for details.

        References
        ----------
        Choulli, T., Elazkany, E., & Vanmaele, M. (2024). The second-order Esscher martingale densities for continuous-time market models. arXiv preprint arXiv:2407.03960.
        """

        
        self.r = r
        self.q = q
        self.K = K
        self.T = T
        self.option_type = option_type
        self.payoff = payoff
        self.emm = emm
        self.psi = psi