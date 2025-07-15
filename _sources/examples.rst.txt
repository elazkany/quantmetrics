Examples
========

This section provides practical examples to help you understand how to use the `quantmetrics` package for option pricing, calibration, and more. These examples are designed to demonstrate the functionality of the core modules and classes.

Calculating Option Prices
-------------------------

The following example demonstrates how to calculate option prices under a Geometric Brownian Motion (GBM) model using the `OptionPricer` class.

.. code-block:: python

    from quantmetrics.levy_models.geometric_brownian_motion import GBM
    from quantmetrics.option_pricing.option_pricer import OptionPricer
    from quantmetrics.option_pricing.option import Option

    # Initialize the GBM model
    gbm = GBM(S0=100, sigma=0.2)

    # Define the option parameters
    option = Option(r=0.05, K=100, T=1.0)

    # Create the option pricer
    pricer = OptionPricer(gbm, option)

    # Calculate the option price using the closed-form method
    price = pricer.closed_form()
    print(f"The calculated option price is: {price:.2f}")

Monte Carlo Simulations
-----------------------

This example demonstrates how to calculate option prices using Monte Carlo simulations.

.. code-block:: python

    from quantmetrics.levy_models.variance_gamma import VarianceGamma
    from quantmetrics.option_pricing.option_pricer import OptionPricer
    from quantmetrics.option_pricing.option import Option

    # Initialize the Variance Gamma model
    vg_model = VarianceGamma(S0=100, sigma=0.25, theta=-0.1, nu=0.2)

    # Define the option parameters
    option = Option(r=0.03, K=90, T=0.5)

    # Create the option pricer
    pricer = OptionPricer(vg_model, option)

    # Calculate the option price using Monte Carlo simulations
    price = pricer.monte_carlo(n_simulations=100000)
    print(f"The Monte Carlo estimated price is: {price:.2f}")

Efficient Frontier Construction
-------------------------------

Here’s how to optimize a portfolio and construct an efficient frontier using `quantmetrics` (if relevant modules exist):

.. code-block:: python

    from quantmetrics.utils.economic_data_downloader import DataLoader
    from quantmetrics.option_pricing.option_pricer2 import OptionPricer2

    # Fetch and preprocess financial data
    data_loader = DataLoader()
    historical_prices = data_loader.download_data('AAPL')

    # Example placeholder for optimization
    # Run optimization or price-related processes using historical data
    print("Efficient frontier example placeholder. Customize for your needs.")
