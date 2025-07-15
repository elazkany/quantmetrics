Price Calculators
=================

The `price_calculators` module contains tools for implementing different pricing models, including closed-form solutions, characteristic functions, and numerical methods.

Modules
-------

The `price_calculators` module is organized into the following submodules:

- **base_calculator.py**:
  Defines the base class for all pricing models. This is not meant to be instantiated directly but serves as the foundation for other calculators.

- **gbm_pricing**:
  Contains classes for the Geometric Brownian Motion (GBM) model. This submodule includes:
  
  - `gbm_calculator.py`: Contains the `GBMCalculator` class for pricing using the GBM model.
  
  - `gbm_closed_form.py`: Provides closed-form solutions for options under the GBM model.
  
  - `gbm_paths_Q.py`: Implements path simulations under the GBM model.

- **cjd_pricing**:
  Implements the Constant Jump Diffusion (CJD) model. This submodule includes:
  
  - `cjd_calculator.py`: Contains the `CJDCalculator` class for pricing using the CJD model.
  
  - `cjd_characteristic_function.py`: Implements the characteristic function for the CJD model.
  
  - `cjd_closed_form.py`: Provides closed-form solutions for options under the CJD model.
  
  - `cjd_paths_Q.py`: Implements path simulations under the CJD model.

- **ljd_pricing**:
  Implements the Lognormal Jump Diffusion (LJD) model. This submodule includes:
  
  - `ljd_calculator.py`: Contains the `LJDCalculator` class for pricing using the LJD model.
  
  - `ljd_characteristic_function.py`: Implements the characteristic function for the LJD model.
  
  - `ljd_closed_form.py`: Provides closed-form solutions for options under the LJD model.
  
  - `ljd_paths_Q.py`: Implements path simulations under the LJD model.

- **dejd_pricing**:
  Provides tools for the Double Exponential Jump Diffusion (DEJD) model. This submodule includes:
  
  - `dejd_calculator.py`: Contains the `DEJDCalculator` class for pricing using the DEJD model.
  
  - `dejd_characteristic_function.py`: Implements the characteristic function for the DEJD model.
  
  - `dejd_closed_form.py`: Provides closed-form solutions for options under the DEJD model.
  
  - `dejd_paths_Q.py`: Implements path simulations under the DEJD model.

- **vg_pricing**:
  Implements the Variance Gamma (VG) model. This submodule includes:
  
  - `vg_calculator.py`: Contains the `VGCalculator` class for pricing using the VG model.
  
  - `vg_characteristic_function.py`: Implements the characteristic function for the VG model.
  
  - `vg_closed_form.py`: Provides closed-form solutions for options under the VG model.


Classes
-------

**Geometric Brownian Motion (GBM)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/price_calculators/gbm
   :nosignatures:

   quantmetrics.price_calculators.gbm_pricing.gbm_calculator.GBMCalculator
   quantmetrics.price_calculators.gbm_pricing.gbm_closed_form.GBMClosedForm
   quantmetrics.price_calculators.gbm_pricing.gbm_characteristic_function.GBMCharacteristicFunction
   quantmetrics.price_calculators.gbm_pricing.gbm_paths_Q.GBMSimulatePathsQ

**Constant Jump Diffusion (CJD)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/price_calculators/cjd
   :nosignatures:

   quantmetrics.price_calculators.cjd_pricing.cjd_calculator.CJDCalculator
   quantmetrics.price_calculators.cjd_pricing.cjd_closed_form.CJDClosedForm
   quantmetrics.price_calculators.cjd_pricing.cjd_characteristic_function.CJDCharacteristicFunction
   quantmetrics.price_calculators.cjd_pricing.cjd_paths_Q.CJDSimulatePathsQ

**Lognormal Jump Diffusion (LJD)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/price_calculators/ljd
   :nosignatures:

   quantmetrics.price_calculators.ljd_pricing.ljd_calculator.LJDCalculator
   quantmetrics.price_calculators.ljd_pricing.ljd_closed_form.LJDClosedForm
   quantmetrics.price_calculators.ljd_pricing.ljd_characteristic_function.LJDCharacteristicFunction
   quantmetrics.price_calculators.ljd_pricing.ljd_paths_Q.LJDSimulatePathsQ

**Double Exponential Jump Diffusion (DEJD)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/price_calculators/dejd
   :nosignatures:

   quantmetrics.price_calculators.dejd_pricing.dejd_calculator.DEJDCalculator
   quantmetrics.price_calculators.dejd_pricing.dejd_closed_form.DEJDClosedForm
   quantmetrics.price_calculators.dejd_pricing.dejd_characteristic_function.DEJDCharacteristicFunction
   quantmetrics.price_calculators.dejd_pricing.dejd_paths_Q.DEJDSimulatePathsQ

**Variance Gamma (VG)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/price_calculators/vg
   :nosignatures:

   quantmetrics.price_calculators.vg_pricing.vg_calculator.VGCalculator
   quantmetrics.price_calculators.vg_pricing.vg_closed_form.VGClosedForm
   quantmetrics.price_calculators.vg_pricing.vg_characteristic_function.VGCharacteristicFunction
   quantmetrics.price_calculators.vg_pricing.vg_paths_Q.VGSimulatePathsQ
