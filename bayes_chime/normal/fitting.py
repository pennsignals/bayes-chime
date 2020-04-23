"""Fitting routines for approximating distributions with normal distributions
"""
from typing import TypeVar, Dict, Any

from numpy import linspace, sqrt

from scipy.optimize import curve_fit
from scipy.stats import norm, beta, gamma

from pandas import DataFrame

from gvar import gvar

from bayes_chime.normal.utilities import FloatLikeArray, NormalDistVar, FloatOrDistVar

Dist = TypeVar("ScipyContinousDistribution")


def fit_norm_dist_to_ens(x: FloatLikeArray) -> NormalDistVar:
    """Approximates ensemble (random vector) by normal distribution
    """
    return gvar(*norm.fit(x))


def fit_norm_dist_to_dist(dist: Dist) -> NormalDistVar:
    """Approximates distribution by normal distribution
    """
    x = linspace(dist.ppf(0.01), dist.ppf(0.99), 100)
    y = dist.pdf(x)

    mu, var = dist.stats(moments="mv")
    mu, std = curve_fit(norm.pdf, xdata=x, ydata=y, p0=(mu, sqrt(var)))[0]

    return gvar(mu, std)


def parse_dist(data: Dict[str, Any]) -> Dist:
    """Parses prior frame data to distribution
    """
    distribution = data["distribution"]
    if distribution == "beta":
        dist = beta(a=data["p1"], b=data["p2"])
    elif distribution == "gamma":
        dist = gamma(a=data["p1"], scale=data["p2"])
    elif distribution == "constant":
        dist = data["base"]
    else:
        raise KeyError(
            "Distribution {distribution} not implemented.".format(
                distribution=distribution
            )
        )
    return dist


def gv_to_dist(normal: NormalDistVar) -> Dist:
    """Converts gvar to scipy dist
    """
    return norm(loc=normal.mean, scale=normal.sdev)


def fit_norm_to_prior_df(prior_df: DataFrame) -> Dict[str, FloatOrDistVar]:
    """Reads in prior data frame (`params.csv`) and returns fitted normal variables.
    """
    priors = {}
    for _, row in prior_df.iterrows():
        dist = parse_dist(row)
        priors[row["param"]] = (  # account for constant dist
            dist if isinstance(dist, float) else fit_norm_dist_to_dist(dist)
        )

    return priors
