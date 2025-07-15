import numpy as np
import pandas as pd
from scipy.optimize import minimize, brute
from typing import Callable, Dict, List, Union, TYPE_CHECKING
from quantmetrics.option_pricing import OptionPricer

if TYPE_CHECKING:
    from quantmetrics.levy_models import LevyModel
    from quantmetrics.option_pricing import Option


class OptionCalibration:
    """
    A class for calibrating model option prices to market option prices

    Parameters
    ----------
    option_data : pd.DataFrame
        DataFrame containing option data with columns ['time_to_maturity', 'market_price','strike_price', 'interest_rate', 'underlying_price'].
    model : object
        Model object that defines the option pricing model.
    option_type : object
        Option type object that defines the type of option (e.g., European, American).
    calibrate_second_Esscher_param : bool, optional
        Flag to calibrate the second Esscher parameter. Default is False.

    Methods
    -------
    rmse_error(params: np.ndarray) -> float:
        Calculate the Root Mean Squared Error (RMSE) between market option prices and model option prices.
    mae_error(params: np.ndarray) -> float:
        Calculate the Mean Absolute Error (MAE) between market option prices and model option prices.
    calibrate(error_function: Callable[[np.ndarray], float], init_params: np.ndarray, method: str = 'Nelder-Mead', brute_search: bool = False, brute_search_tuple: tuple = None, tol: float = 1e-6) -> dict:
        Calibrate the model to the market data by minimizing the specified error function.

    Returns
    -------
    dict
        Dictionary containing the optimization result with keys ['message', 'success', 'status', 'fun', 'params', 'nit', 'nfev', 'final_simplex'].

    References
    ----------
    Hilpisch, Y. (2015). Derivatives analytics with Python: data analysis, models, simulation, calibration and hedging. John Wiley & Sons.

    Examples
    --------

    """

    def __init__(
        self,
        option_data: pd.DataFrame,
        model: "LevyModel",
        option: "Option",
        calibrate_second_Esscher_param: bool = False,
    ):
        self.option_data = option_data
        self.model = model
        self.option = option
        self.calibrate_psi = calibrate_second_Esscher_param

        # calibration for fixed T and r with an array of strike prices
        self.option.K = self.option_data["strike_price"].values
        self.option.T = self.option_data["time_to_maturity"].values[0]  # annualized
        self.option.r = self.option_data["interest_rate"].values[0]  # annualized
        self.option.emm = option.emm

        self.market_price = self.option_data["market_price"].values

        self.model.S0 = self.option_data["underlying_price"].values[0]

        self.i = 0  # counter initialization
        self.min_error = 100  # minimal error initialization

    @property
    def params_to_be_calibrated(self):
        return self._get_params_to_be_calibrated()

    def _get_params_to_be_calibrated(self):
        if self.option.emm == "mean-correcting":
            return {k: v for k, v in self.model.model_params.items() if k != "mu"}
        elif (self.option.emm == "Esscher") & (self.calibrate_psi == False):
            return self.model.model_params.copy()
        elif (self.option.emm == "Esscher") & (self.calibrate_psi == True):
            cal_params = self.model.model_params.copy()
            cal_params["psi"] = self.option.psi
            return cal_params
        else:
            raise ValueError("Unknown EMM type: {}".format(self.option.emm))

    def rmse_error(self, params: np.ndarray) -> float:
        """
        Calculate the Root Mean Squared Error (RMSE) between market option prices and model option prices.

        Parameters
        ----------
        params : np.ndarray
            Model parameters.

        Returns
        -------
        float
            RMSE value.
        """

        # create a dictionary of the parameters being calibrated
        new_params = {
            key: value
            for key, value in zip(self.params_to_be_calibrated.keys(), params.tolist())
        }

        # For the given model, change the values of the parameters
        for key, value in new_params.items():
            self.model.model_params[key] = value
            setattr(self.model, key, value)

        # check the new parameters satisfy parameters conditions
        if not self.model.model_params_conds_valid:
            return 500.0

        # calculate the option price using the new parameters
        option_pricer = OptionPricer(self.model, self.option)
        model_price = option_pricer.fft()

        # calculate RMSE
        RMSE = np.sqrt(np.mean((model_price - self.market_price) ** 2))

        self.min_error = min(self.min_error, RMSE)
        if self.i % 50 == 0:
            print(
                "%4d |" % self.i,
                new_params,
                "| RMSE = %7.3f | min error = %7.3f" % (RMSE, self.min_error),
            )
        self.i += 1
        return RMSE

    def mae_error(self, params: np.ndarray) -> float:
        """
        Calculate the Mean Absolute Error (RMSE) between market option prices and model option prices.

        Parameters
        ----------
        params : np.ndarray
            Model parameters.

        Returns
        -------
        float
            MAE value.
        """

        # create a dictionary of the parameters being calibrated
        new_params = {
            key: value
            for key, value in zip(self.params_to_be_calibrated.keys(), params.tolist())
        }

        # For the given model, change the values of the parameters
        for key, value in new_params.items():
            self.model.model_params[key] = value
            setattr(self.model, key, value)

        # check the new parameters satisfy parameters conditions
        if not self.model.model_params_conds_valid:
            return 500.0

        # calculate the option price using the new parameters
        option_pricer = OptionPricer(self.model, self.option)
        model_price = option_pricer.fft()

        # calculate MAE
        MAE = np.mean(np.abs(model_price - self.market_price))

        self.min_error = min(self.min_error, MAE)
        if self.i % 50 == 0:
            print(
                "%4d |" % self.i,
                new_params,
                "| MAE = %7.3f | min error = %7.3f" % (MAE, self.min_error),
            )
        self.i += 1
        return MAE

    def calibrate(
        self,
        error_function: Callable[[np.ndarray], float],
        init_params: np.ndarray,
        method: str = "Nelder-Mead",
        brute_search: bool = False,
        brute_search_tuple: tuple = None,
        tol: float = 1e-6,
    ) -> Dict[str, Union[str, float, bool, int, Dict[str, Union[float, int]]]]:
        """
        Calibrate the model to the market data by minimizing the specified error function.

        Parameters
        ----------
        error_function : Callable[[np.ndarray], float]
            The error function to be minimized.
        init_params : np.ndarray
            Initial model parameters.
        method : str, optional
            The optimization method to use. Default is 'Nelder-Mead'.
        brute_search : bool, optional
            If True, perform a brute search for the parameters. Default is False.
        brute_search_tuple : tuple, optional
            Tuple of parameter ranges for brute search. Each tuple contains (start, end, step) values.
        tol : float, optional
            Tolerance for termination. Default is 1e-6.

        Returns
        -------
        dict
            Dictionary containing the optimization result with keys ['message', 'success', 'status', 'fun', 'params', 'nit', 'nfev', 'final_simplex'].
        """

        if brute_search == True and brute_search_tuple is not None:
            init_params = brute(
                error_function,
                brute_search_tuple,
                finish=None,
            )

        result = minimize(
            error_function, init_params, method=method, tol=tol
        )  # fmin(error_function, init_params, maxiter=500,maxfun=750, xtol=0.000001, ftol = 0.000001)

        opt_params = {
            key: value
            for key, value in zip(
                self.params_to_be_calibrated.keys(), result.x.tolist()
            )
        }

        return {
            "message": result.message,
            "success": result.success,
            "status": result.status,
            "fun": result.fun,
            "params": opt_params,
            "nit": result.nit,
            "nfev": result.nfev,
            "final_simplex": result.final_simplex,
        }
