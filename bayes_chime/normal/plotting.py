"""Plotting routines
"""
from typing import TypeVar, Dict, Any

from numpy import linspace

from matplotlib.pylab import gca as get_current_actor
from seaborn import distplot

from gvar import mean as gv_mean
from gvar import sdev as gv_sdev

from bayes_chime.normal.utilities import NormalDistVar
from bayes_chime.normal.fitting import (
    fit_norm_dist_to_dist,
    parse_dist,
    gv_to_dist,
    fit_norm_dist_to_ens,
)

Axes = TypeVar("Axes")


def plot_prior_fit(**kwargs):
    """Parses distribution from meta parameters and plots exact distribution and
    normal fit
    """
    data = kwargs["data"].iloc[0].to_dict()
    dist = parse_dist(data)

    ax = get_current_actor()

    x = linspace(dist.ppf(0.001), dist.ppf(0.999), 100)
    y = dist.pdf(x)
    ax.fill_between(x, y, alpha=0.5)

    plot_gv_dist(fit_norm_dist_to_dist(dist), color="black")


def plot_posterior_fit(**kwargs):
    """Fits normal distribution to ensemble and plots normal dist as well as hist
    """
    ax = get_current_actor()
    x = kwargs["data"].x.values
    distplot(a=x, kde=False, ax=ax, hist_kws={"density": True})
    plot_gv_dist(
        fit_norm_dist_to_ens(x, thresh=kwargs.get("thresh", None)), ax=ax, color="black"
    )


def plot_gv_dist(gvar: NormalDistVar, **kwargs):
    """Plots pdf of gvar
    """
    ax = kwargs.pop("ax", get_current_actor())

    normal = gv_to_dist(gvar)
    x = linspace(normal.ppf(0.001), normal.ppf(0.999), 100)
    y_fit = normal.pdf(x)
    ax.plot(x, y_fit, **kwargs)

    return ax


def plot_gvar(
    line_kws: Dict[str, Any] = None, fill_kws: Dict[str, Any] = None, **kwargs
) -> Axes:
    """Plots gvar as a band.

    Arguments:
        line_kws: Kwargs specific for line plot
        fill_kws: Kwargs specific for fill between
        kwargs:  Shared kwargs
            Requires: x and y
            Optional: z_factor to upadte band with (e.g., 0.674 for 0.5 CI)
    """
    y = kwargs.pop("y")
    x = kwargs.pop("x")
    yy_mean = gv_mean(y)
    yy_sdev = gv_sdev(y)

    z_factor = kwargs.pop("z_factor", 1)
    yy_sdev *= z_factor

    return plot_band(
        x=x,
        y1=yy_mean - yy_sdev,
        ym=yy_mean,
        y2=yy_mean + yy_sdev,
        line_kws=line_kws,
        fill_kws=fill_kws,
        **kwargs,
    )


def plot_band(
    line_kws: Dict[str, Any] = None, fill_kws: Dict[str, Any] = None, **kwargs
) -> Axes:
    """Plots gvar as a band.

    Arguments:
        line_kws: Kwargs specific for line plot
        fill_kws: Kwargs specific for fill between
        kwargs:  Shared kwargs
            Requires: x and y1, ym, y2
    """
    line_kws = line_kws.copy() or {}
    fill_kws = fill_kws.copy() or {}

    y1 = kwargs.pop("y1")
    ym = kwargs.pop("ym")
    y2 = kwargs.pop("y2")
    x = kwargs.pop("x")

    ax = kwargs.pop("ax", get_current_actor())

    line_kws.update(kwargs)
    fill_kws.update(kwargs)

    ax.plot(x, ym, **line_kws)
    ax.fill_between(x, y1, y2, **fill_kws)

    return ax
