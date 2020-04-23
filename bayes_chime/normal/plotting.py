"""Plotting routines
"""
from numpy import linspace

from matplotlib.pylab import gca as get_current_actor


from bayes_chime.normal.fitting import fit_norm_dist_to_dist, parse_dist, gv_to_dist


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

    normal = gv_to_dist(fit_norm_dist_to_dist(dist))
    x = linspace(normal.ppf(0.001), normal.ppf(0.999), 100)
    y_fit = normal.pdf(x)
    ax.plot(x, y_fit, color="black")
