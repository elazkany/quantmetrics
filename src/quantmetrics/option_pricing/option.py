#option_pricing/option.py

class Option:
    def __init__(
            self,
            r: float = 0.05/365,
            q: float = 0,
            K: float = 60,
            T: float = 63,
            payoff: str = "c",
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
        """

        
        self.r = r
        self.q = q
        self.K = K
        self.T = T
        self.payoff = payoff