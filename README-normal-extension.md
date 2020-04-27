# Normal extension of Bayesian analysis

The folder `bayes_chime/normal` contains an extension of the Bayesian analysis of the SEIR model.
In this extension, it is assumed that all data and model parameters are normally distributed.
This simplifies the propagation of errors and allows to analytically compute approximated posterior distributions.
Particularly, propagating parameter distributions through a 200-day simulation takes `20ms` while the estimation of the posterior distribution is at the order of `1s`.
Tests indicated that, for a given parameter distribution, general Bayesian forecasts agree with their normal approximations within uncertainty.


## Install the module
You can locally install this module using pip

```bash
pip install [--user] [-e] .
```

## How to use the module

### How to propagate uncertainties
```python
from gvar import gvar
from bayes_chime.normal.models import SEIRModel

# define fixed model parameters
xx = {"market_share": 0.26, "region_pop": 1200000, ...}

# define model parameter distributions
pp = {"incubation_days": gvar(4.6, 1.5), "recovery_days": gvar(15.3, 3.5)}

# set up the model
seir = SEIRModel()
df = seir.propagate_uncertainties(xx, pp)
```
The `gvar` variable represents a Gaussian random number characterized by it's mean and standard deviation.

### How to compute posteriors
```python
... # after code above

from lsqfit import nonlinear_fit

yy = gvar(data_mean, data_sdev)

fit = nonlinear_fit(data=(xx, yy), prior=pp, fcn=seir.fit_fcn)

print(fit) # Fit statistics
print("Posterior:", fit.p)
```

In the `notebooks/How-to-use-normal-approximations-module.ipynb`, the general usage of the module is explained.

## Technical details

### Uncertainty propagation and posteriors

This module makes use of two libraries developed by [Peter Lepage](https://physics.cornell.edu/peter-lepage)

* [`gvar`](https://gvar.readthedocs.io) and
* [`lsqfit`](https://lsqfit.readthedocs.io)

The random numbers represented by `gvar`s act like `floats` but automatically compute analytic derivatives in successive computations.
Thus they allow to propagate errors through the whole computation.
Because normal distributions are self-conjugate (if the likelihood is a Gaussian and the prior is Gaussian, so will be the posterior).

To determine posterior distributions, the `nonlinear_fit` function from `lsqfit` uses the saddle-point approximation for the kernel of the posterior function (evolve the residuals up to the second order and evaluate at the point where the first derivative vanishes).
Thus, computing the posterior effectively boils down to finding the parameters which cause the first derivative of the `chi**2` to vanish.
For this it uses regular minimization routines.
More details are specified in the [appendix A of the linked publication](https://arxiv.org/pdf/1406.2279.pdf).

### Kernel functions

To utilize the above modules, kernel operations must be written to utilize the syntax of `gvar`s and `lsqfit`.
Model parameters are either categorized as independent / fixed parameters or distribution / variable parameters.
Variable parameters follow normal distributions, fixed parameters are numbers.

This module abstracts the compartment models like SIR or SEIR such that future implementations can easily be extended.
E.g., after inheriting from the `bayes_chime.normal.models.base.CompartmentModel`, one only has to provide a `simulation_step` method and can use existing API to run simulations
```python
def simulation_step(data, **pars):
    susceptible = data["susceptible"]
    exposed = data["exposed"]
    infected = data["infected"]
    recovered = data["recovered"]

    infected += pars["beta"] * infected * susceptible / pars["total"]
    ...

    return oupdated_data
```

```

## Cross checks

This module tests against the non-Bayesian `penn_chime` module. These tests can be run in the repo root with
```bash
pytest
```
Furthermore, the `How-to-use` notebook fits posterior distributions generated with the main module and compares the propagation of uncertainties.
