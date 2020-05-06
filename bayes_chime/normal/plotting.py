"""Plotting routines
"""
from typing import TypeVar, Dict, Any, Optional, List

from numpy import linspace, array, where
from pandas import DataFrame

from matplotlib.pylab import gca as get_current_actor
from matplotlib.pylab import subplots
from seaborn import distplot

from gvar import mean as gv_mean
from gvar import sdev as gv_sdev

from bayes_chime.normal.utilities import NormalDistVar, NormalDistArray
from bayes_chime.normal.fitting import (
    fit_norm_dist_to_dist,
    parse_dist,
    gv_to_dist,
    fit_norm_dist_to_ens,
)

Axes = TypeVar("Axes")
Figure = TypeVar("Figure")


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
    line_kws: Dict[str, Any] = None,
    fill_kws: Dict[str, Any] = None,
    y_min: Optional[float] = None,
    **kwargs
) -> Axes:
    """Plots gvar as a band.

    Arguments:
        line_kws: Kwargs specific for line plot
        fill_kws: Kwargs specific for fill between
        y_min: Minimal value for data
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
        y_min=y_min,
        **kwargs,
    )


def plot_band(
    line_kws: Dict[str, Any] = None,
    fill_kws: Dict[str, Any] = None,
    y_min: Optional[float] = None,
    **kwargs
) -> Axes:
    """Plots gvar as a band.

    Arguments:
        line_kws: Kwargs specific for line plot
        fill_kws: Kwargs specific for fill between
        y_min: Minimal value for data
        kwargs:  Shared kwargs
            Requires: x and y1, ym, y2
    """
    line_kws = line_kws.copy() or {}
    fill_kws = fill_kws.copy() or {}

    y1 = kwargs.pop("y1")
    ym = kwargs.pop("ym")
    y2 = kwargs.pop("y2")
    if y_min is not None:
        y1 = where(y1 < y_min, y_min, y1)
        ym = where(ym < y_min, y_min, ym)
    x = kwargs.pop("x")

    ax = kwargs.pop("ax", get_current_actor())

    line_kws.update(kwargs)
    fill_kws.update(kwargs)

    ax.plot(x, ym, **line_kws)
    ax.fill_between(x, y1, y2, **fill_kws)

    return ax


def plot_fit(  # pylint: disable=R0914
    fit_df: DataFrame,
    columns: List[List[str]],
    data: Optional[Dict[str, NormalDistArray]] = None,
) -> Figure:
    """Creates a grid plot for fitted model

    Arguments:
        fit_df: The CompartmentModel prediction to plot
        columns: The columns to plot aranged as the desired grid
        data: If present, matches col name against data and plots in the same frame.
            Assumes data has the same time values as the corresponding fit_df but end
            earlier.
    """
    nrows, ncols = array(columns).shape

    data = data or {}

    fig, axs = subplots(
        ncols=ncols, nrows=nrows, figsize=(8 * ncols, 5 * nrows), sharex=True
    )

    gv_kws = {"zorder": 10, "lw": 3}
    gv_line_kws = {"ls": "--"}
    gv_fill_kws = {"alpha": 0.2}

    for ir, (row, ax_row) in enumerate(zip(columns, axs)):
        for ic, (col, ax) in enumerate(zip(row, ax_row)):
            name = col.replace("_", " ").capitalize()

            for ci_label, (z_fact, alpha) in {
                "50% CI": (0.674, 0.2),
                "90% CI": (1.645, 0.1),
            }.items():
                plot_gvar(
                    x=fit_df.index,
                    y=fit_df[col].values,
                    y_min=0,
                    ax=ax,
                    **gv_kws,
                    z_factor=z_fact,
                    color="black",
                    line_kws={**gv_line_kws},
                    fill_kws={"alpha": alpha, "label": "Fit " + ci_label},
                )

            if col in data:
                plot_gvar(
                    x=fit_df.index[: len(data[col])],
                    y=data[col],
                    y_min=0,
                    ax=ax,
                    color="red",
                    line_kws={**gv_line_kws, "label": "Data"},
                    fill_kws={**gv_fill_kws, "alpha": 0.5, "zorder": 5},
                )

            ax.set_ylabel(name)
            ax.grid(True)
            if ic == 0 == ir:
                ax.legend(loc="upper left")

    fig.suptitle("Normal PDF at 50% C.I.", y=1.02, fontsize=12, fontweight="bold")
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig
