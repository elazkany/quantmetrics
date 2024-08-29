import pytest
from quantmetrics.levy_models import GBM
from quantmetrics.option_pricing import Option, OptionPricer

gbm = GBM()

option = Option()

pricer = OptionPricer(model = gbm, option = option)

pricer.exact() 

pricer.fft()

pricer.monte_carlo()