import numpy as np
from quantmetrics.levy_models.geometric_brownian_motion import GeometricBrownianMotion

def test_fit_gbm_estimates_close_to_true():
    np.random.seed(42)
    mu, sigma = 0.05, 0.2
    data = np.random.normal(mu - 0.5 * sigma**2, sigma, size=1000)

    model = GeometricBrownianMotion()
    result = model.fit(data)

    assert result.success
    assert abs(model._mu - mu) < 0.02
    assert abs(model._sigma - sigma) < 0.02