# tests/test_gbm_bsm_mle.py

# import pytest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quantmetrics.levy_models import GBM, CJD, LJD

from quantmetrics.option_pricing import Option, OptionPricer

from quantmetrics.utils import load_data

gbm = GBM()

option = Option()

pricer = OptionPricer(model=gbm, option=option)

pricer.exact()

pricer.fft()

pricer.monte_carlo()


wti_data = load_data('df_wti')

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

mle_gbm_params = mle_gbm.x

cjd = CJD()

option = Option(emm = "Esscher", psi = 1700.0)

pricer = OptionPricer(model=cjd, option=option)

pricer.exact()

pricer.fft()

pricer.monte_carlo(num_paths = 200000)

ljd = LJD()#sigmaJ = 0)
option = Option()

pricer = OptionPricer(model=ljd, option=option)

pricer.exact()
pricer.fft()

pricer.monte_carlo(num_timesteps=200, num_paths=200000)

