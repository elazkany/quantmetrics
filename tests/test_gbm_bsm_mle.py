# tests/test_gbm_bsm_mle.py

# import pytest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quantmetrics.levy_models import GBM, CJD, LJD

from quantmetrics.option_pricing import Option, OptionPricer

from quantmetrics.utils import load_data


wti_data = load_data("df_wti")

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

gbm = GBM()

mle_gbm = gbm.fit(logreturns)

mle_gbm_params = mle_gbm.x

cjd = CJD()

mle_cjd = cjd.fit(logreturns)

mle_cjd_params = mle_cjd.x

ljd = LJD()

mle_ljd = ljd.fit(logreturns)

mle_ljd_params = mle_ljd.x


# Create histogram
num_bins = 100

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(
    logreturns, num_bins, density=True, color="#fffafa", edgecolor="grey"
)

# add a 'best fit' line
gbm_density = gbm.pdf(data=bins, est_params=mle_gbm_params)

cjd_density = cjd.pdf(data=bins, est_params=mle_cjd_params)

ljd_density = ljd.pdf(data=bins, est_params=mle_ljd_params)

heights, bin = np.histogram(logreturns, bins=len(list(set(logreturns))))

percent = [i / sum(heights) * 100 for i in heights]

# ax.bar(bin[:-1], percent, width=2500, align="edge")

vals = ax.get_yticks()

ax.set_yticklabels(["%1.2f%%" % i for i in vals])

ax.plot(bins, gbm_density, "b", label="GBM")
ax.plot(bins, cjd_density, "r", label="CJD")
ax.plot(bins, ljd_density, "g", label="LJD")
ax.set_xlabel("Value")
ax.set_ylabel("Probability density")
ax.set_title("Compare empirical and estimated densities")
ax.legend(loc="upper left", fontsize="large")

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()

# pricing

option = Option()

pricer = OptionPricer(model=gbm, option=option)

pricer.exact()

pricer.fft()

pricer.monte_carlo()[0]

cjd = CJD()

option = Option(emm="Esscher", psi=50000.0)

pricer = OptionPricer(model=cjd, option=option)

pricer.exact()

pricer.fft()

pricer.monte_carlo(num_paths=200000)

ljd = LJD()  # sigmaJ = 0)
option = Option()

pricer = OptionPricer(model=ljd, option=option)

pricer.exact()
pricer.fft()

pricer.monte_carlo(num_timesteps=200, num_paths=200000)
