# Quantmetrics

![Documentation Deployment](https://github.com/elazkany/quantmetrics/actions/workflows/deploy_docs.yaml/badge.svg)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://elazkany.github.io/quantmetrics/)

## 🔍 Project Overview

Quantmetrics is a Python package for option pricing and model calibration, offering a modular, extensible framework for conducting financial computations and numerical experiments. It leverages object-oriented programming to model Lévy processes and perform pricing using both established and novel methodologies.

> 🚧 This package is a work in progress. It currently supports a subset of models and European-style options, with future development planned for additional models, calibration techniques, and option types (e.g., American options).

## 📦 Installation

The quantmetrics package can be installed directly from GitHub:

```bash
pip install git+https://github.com/elazkany/quantmetrics.git
```

## ⚡ Quick Start

Below is a basic example demonstrating how to price European options using the Lognormal Jump Diffusion (LJD) model under both the classical and second-order Esscher pricing frameworks.

```python
import numpy as np
from quantmetrics.levy_models import LJD
from quantmetrics.option_pricing import Option, OptionPricer

# --- Classical Esscher Pricing (psi = 0) ---

# Initialize the LJD model with default parameters
ljd = LJD()  # S0=50, sigma=0.2, lambda_=1, muJ=-0.1, sigmaJ=0.1

# Define a European call option
option = Option(K=50, T=20/252)  # r=0.05, q=0.02, psi=0

# Create the pricer and compute the option price using FFT
pricer = OptionPricer(model=ljd, option=option)
price_classical = pricer.fft()


# --- Second-Order Esscher Pricing (psi ≠ 0) ---

# Reuse the same model instance or create a new one
ljd = LJD()

# Define the same option but with a second-order Esscher transform (psi = 40)
option = Option(K=50, T=20/252, psi=40, emm="Esscher")

# Compute price under the second-order framework
pricer = OptionPricer(model=ljd, option=option)
price_second_order = pricer.fft()

```

## 📄 Documentation

You can find the full documentation here:

👉 [documentation](https://elazkany.github.io/quantmetrics)

## 🔬 Research Background

This package was developed as part of a research project on second-order Esscher pricing, conducted under the supervision of Prof. Tahir Choulli and in collaboration with Prof. Michèle Vanmaele. The project aimed to extend classical option pricing techniques and create a modular framework for testing and validating new methodologies.

## 📚 References

Choulli, T., Elazkany, E., & Vanmaele, M. (2024). The second-order Esscher martingale densities for continuous-time market models. arXiv preprint arXiv:2407.03960.

Choulli, T., Elazkany, E., & Vanmaele, M. (2024). Applications of the Second-Order Esscher Pricing in Risk Management. arXiv preprint arXiv:2410.21649.