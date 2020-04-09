from scipy.optimize import fmin
from scipy.stats import gamma, beta
import numpy as np


def gamma_from_q(l, u, quantiles_percent=0.95):
    def loss(params):
        a, b = params
        lq = (1 - quantiles_percent) / 2
        uq = 1 - lq
        return ( (gamma.cdf(l, a, scale=b) - lq)**2 + (gamma.cdf(u, a, scale=b) - uq)**2 )

    start_params = (5, 5)
    fit = fmin(loss, start_params, disp = 0)
    return fit


def beta_from_q(l, u, quantiles_percent=0.95):
    def loss(params):
        a, b = params
        lq = (1 - quantiles_percent) / 2
        uq = 1 - lq
        return ( (beta.cdf(l, a, b) - lq)**2 + (beta.cdf(u, a, b) - uq)**2 )

    start_params = (1, 1)
    fit = fmin(loss, start_params, disp = 0)
    return fit

# # Usage:
# # Let's say I want my beta prior have 90% of it's probability mass
# # between 0.3 and 0.5:
# params = beta_from_q(0.3, 0.5)

# # Let's see wat that distribution looks like:
# x = np.linspace(0, 1, 100)
# fig, ax = plt.subplots(1, 1)
# ax.plot(x, beta.pdf(x, params[0], params[1]))
# ax.axvline(0.3, ls='--')
# ax.axvline(0.5, ls='--')

# # Let's say I want my gamma prior have 90% of it's probability mass
# # between 0.3 and 0.5:
# params = gamma_from_q(2.5, 6)

# # Let's see wat that distribution looks like:
# x = np.linspace(0, 10, 100)
# fig, ax = plt.subplots(1, 1)
# ax.plot(x, gamma.pdf(x, params[0], scale=params[1]))
# ax.axvline(2.5, ls='--')
# ax.axvline(6, ls='--')
