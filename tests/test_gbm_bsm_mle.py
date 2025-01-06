# tests/test_gbm_bsm_mle.py

# import pytest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quantmetrics.levy_models import GBM, CJD, LJD

from quantmetrics.option_pricing import Option, OptionPricer, RiskPremium

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


# Nov 18 testing the Merton M.E with Esscher


import numpy as np
import matplotlib.pyplot as plt
from quantmetrics.levy_models import GBM, CJD, LJD

from quantmetrics.option_pricing import Option, OptionPricer, RiskPremium




gbm = GBM()
cjd = CJD()
ljd = LJD(muJ=cjd.gamma, sigmaJ=0)

option = Option(emm="Black-Scholes")

option = Option(emm="Esscher", psi=0)

gbm_price = OptionPricer(model=gbm, option=Option(emm="Black-Scholes"))
cjd_price = OptionPricer(model=cjd, option=option)
ljd_price = OptionPricer(model=ljd, option=option)

gbm_price.exact()
gbm_price.fft()
gbm_price.monte_carlo(num_timesteps=200, num_paths=100000)
cjd_price.exact()
cjd_price.fft()
cjd_price.monte_carlo(num_timesteps=200, num_paths=100000)
ljd_price.exact()
ljd_price.fft()
ljd_price.monte_carlo(num_timesteps=200, num_paths=100000)




psi = np.arange(-100, 230, 5) #50, 50)

thetas = np.array([])
thetas2 = np.array([])
cjd_prices_exact = np.array([])
cjd_prices_fft = np.array([])
cjd_prices_mc = np.array([])
cjd_prices_mc = np.array([])
ljd_prices_exact = np.array([])
ljd_prices_fft = np.array([])
ljd_prices_mc = np.array([])
ljd_mc_error = np.array([])
cjd_mc_error = np.array([])

for i in range(0, len(psi)):
    cjd = CJD()
    ljd = LJD(muJ=cjd.gamma, sigmaJ=0.01)
    option = Option(emm="Esscher", psi=psi[i])
    ljd_price = OptionPricer(model=ljd, option=option)
    thetas = np.append(thetas, RiskPremium(ljd, option).calculate())
    ljd_prices_exact = np.append(ljd_prices_exact, ljd_price.exact())
    ljd_prices_fft = np.append(ljd_prices_fft, ljd_price.fft())
    ljd_temp = ljd_price.monte_carlo(num_timesteps=200, num_paths=10000)
    ljd_prices_mc = np.append(
        ljd_prices_mc, ljd_temp[0]
    )
    ljd_mc_error = np.append(
        ljd_mc_error, ljd_temp[1]
    )


    cjd_price = OptionPricer(model=cjd, option=option)
    thetas2 = np.append(thetas2, RiskPremium(cjd, option).calculate())
    cjd_prices_exact = np.append(cjd_prices_exact, cjd_price.exact())
    cjd_prices_fft = np.append(cjd_prices_fft, cjd_price.fft())
    cjd_temp = cjd_price.monte_carlo(num_timesteps=200, num_paths=10000)
    cjd_prices_mc = np.append(
        cjd_prices_mc, cjd_temp[0]
    )
    cjd_mc_error = np.append(
        cjd_mc_error, cjd_temp[1]
    )

plt.plot(psi[0: ], cjd_prices_fft[0: ], label="cjd fft")
plt.plot(psi[0: ], ljd_prices_fft[0: ], label="ljd fft")
#plt.plot(psi[0: ], ljd_prices_exact[0: ], label="ljd exact")
plt.plot(psi[0: ], cjd_prices_exact[0: ], label="cjd exact")
plt.plot(psi[0: ], ljd_prices_mc[0: ], label="ljd MC")
plt.plot(psi[0: ], cjd_prices_mc[0: ], label="cjd MC")
plt.title("Comparison of LJD Prices")
plt.xlabel("Psi")
plt.ylabel("Prices")
plt.legend()  # Adding the legend
plt.show()


# MC error
plt.plot(psi[0: ], ljd_mc_error[0: ], label="ljd MC error")
plt.plot(psi[0: ], cjd_mc_error[0: ], label="cjd MC error")
plt.title("Comparison of MC standard errors")
plt.xlabel("Psi")
plt.ylabel("Error")
plt.legend()  # Adding the legend
plt.show()

# Nov 25

psi = np.arange(-500, 220, 1)

thetas = np.array([])

for i in range(0, len(psi)):
    ljd = LJD()
    #ljd.mu = .2
    #ljd.sigma =.2
    #ljd.lambda_ = .2
    #ljd.muJ = .2
    #ljd.sigmaJ=.2
    
    option = Option(emm="Esscher", psi=psi[i])
    thetas = np.append(thetas, RiskPremium(ljd, option).calculate())

psi_cond = 1 / (2 * ljd.sigmaJ**2)
theta_point =  -ljd.muJ / (ljd.sigmaJ**2)

plt.plot(psi, thetas, label="theta")
plt.axvline(x=psi_cond, color='r', linestyle='--', linewidth=2)
#plt.axhline(y=theta_point, color='g', linestyle='--', linewidth=2)
plt.title("martingale equation")
plt.xlabel("Psi")
plt.ylabel("theta")
plt.legend()  # Adding the legend
plt.show()


