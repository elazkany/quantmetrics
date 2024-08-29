import pytest

import numpy as np
import pandas as pd

from quantmetrics.levy_models import GBM

from quantmetrics.option_pricing import Option, OptionPricer

gbm = GBM()

option = Option()

pricer = OptionPricer(model=gbm, option=option)

pricer.exact()

pricer.fft()

pricer.monte_carlo()


wti_data = pd.read_excel(
    "H:\My Drive\Projects\LatexProjects\Latex_projects\Esscher_application\data\RWTCd.xls",
    sheet_name="Data 1",
)

wti_data["date"] = pd.to_datetime(wti_data["date"])

wti_data = wti_data[wti_data["date"] <= "2010-08-31"]

wti_data.columns = ["date", "price"]

logreturns = np.diff(
    np.log(
        wti_data[["price"]].values.reshape(
            wti_data.shape[0],
        )
    )
)

mle_gbm = gbm.fit(logreturns)

mle_gbm