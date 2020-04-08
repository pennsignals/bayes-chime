


import pandas as pd
import os
from _99_shared_functions import *
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy import stats as sps

import sys
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

datadir = f'{os.getcwd()}/data/'
outdir = f'{os.getcwd()}/output/'
figdir = f'{os.getcwd()}/figures/'

# import the census time series and set the zero day to be the first instance of zero
for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}: {arg}")

hospital = sys.argv[1]
census_ts = pd.read_csv(f"{datadir}{hospital}_ts.csv")
first_day = census_ts['date'].values[0]
params = pd.read_csv(f"{datadir}{hospital}_parameters.csv")
# impute vent with the proportion of hosp.  this is a crude hack
census_ts.loc[census_ts.vent.isna(), 'vent'] = census_ts.hosp.loc[census_ts.vent.isna()]*np.mean(census_ts.vent/census_ts.hosp)

nobs = census_ts.shape[0]

# define capacity
vent_capacity = float(params.base.loc[params.param == 'vent_capacity'])
hosp_capacity = float(params.base.loc[params.param == 'hosp_capacity'])

df = pd.read_pickle(f'{outdir}{hospital}_chains.pkl')

# remove burnin
df = df.loc[(df.iter>1000)] #& (~df.chain.isin([1, 12]))]


# plot of logistic curves
def logistic(L, k, x0, x):
    return L/(1+np.exp(-k*(x-x0)))

qlist = []
for day in range(census_ts.shape[0]):
    ldist = logistic(df.logistic_L, 
                      df.logistic_k,
                      df.logistic_x0 - df.offset.astype(int),
                      day)
    qlist.append(np.quantile(ldist, [.05, .5, .95]))


# logistic SD plot
qmat = np.vstack(qlist)
fig = plt.figure()

plt.plot(list(range(census_ts.shape[0])), 1-qmat[:,1])
plt.fill_between(x = list(range(census_ts.shape[0]))
                 ,y1=1-qmat[:,0]
                 ,y2 = 1-qmat[:,2]
                 ,alpha=.3
                 ,lw=2
                 ,edgecolor='k'
               )
plt.ylabel(f'Relative (effective) social contact')
plt.xlabel(f'Days since {first_day}')
fig.savefig(f"{figdir}{hospital}_effective_soc_dist.pdf")




# plot of chains
def plt_predictive(howfar=200):
    plt.scatter(x = df.groupby(['chain']).mean()[['posterior']], y = df.groupby(['chain']).mean()[['doubling_time']])
    plt.ylabel(f'Doubling time mean by chain')
    plt.xlabel(f'Posterior mean by chain')

    # predictive plot
    arrs = np.stack([df.arr.iloc[i] for i in range(df.shape[0])])
    arrq = np.quantile(arrs, axis = 0, q = [.05, .25, .5, .75, .95])

    fig, ax = plt.subplots(figsize=(16, 8), ncols=2, nrows=1)
    # hosp
    ax[0].plot(arrq[2,:howfar,3], label = 'posterior median')
    ax[0].set_ylabel(f'Hospital census', fontsize=12, fontweight='bold')
    ax[0].set_xlabel(f'Days since {first_day}', fontsize=12, fontweight='bold')
    ax[0].fill_between(x = list(range(howfar)),
                       y1 = arrq[0,:howfar,3],
                       y2 = arrq[4,:howfar,3], 
                       label = '90% Credible Region',
                       alpha = .1,
                       lw = 2,
                       edgecolor = "k")
    ax[0].fill_between(x = list(range(howfar)),
                       y1 = arrq[1,:howfar,3],
                       y2 = arrq[3,:howfar,3], 
                       label = '50% Credible Region',
                       alpha = .1,
                       lw = 2,
                       edgecolor = "k")
    ax[0].plot(list(range(len(census_ts.hosp))), census_ts.hosp, 
               color = "red",
               label = "observed")
    ax[0].axhline(y=hosp_capacity, color='k', ls='--', label = "hospital capacity")

    ax[0].legend()



    ax[1].plot(arrq[2,:howfar,5], label = 'posterior median')
    ax[1].set_ylabel(f'Vent census', fontsize=12, fontweight='bold')
    ax[1].set_xlabel(f'Days since {first_day}', fontsize=12, fontweight='bold')
    ax[1].fill_between(x = list(range(howfar)),
                       y1 = arrq[0,:howfar,5],
                       y2 = arrq[4,:howfar,5], 
                       label = '90% Credible Region',
                       alpha = .1,
                       lw = 2,
                       edgecolor = "k",)
    ax[1].fill_between(x = list(range(howfar)),
                       y1 = arrq[1,:howfar,5],
                       y2 = arrq[3,:howfar,5], 
                       label = '50% Credible Region',
                       alpha = .1,
                       lw = 2,
                       edgecolor = "k")
    ax[1].axhline(y=vent_capacity, color='k', ls='--', label = "vent capacity")
    ax[1].plot(list(range(len(census_ts.vent))), census_ts.vent, 
               color = "red",
               label = "observed")
    ax[1].legend()
    fig.savefig(f"{figdir}{hospital}_forecast_{howfar}_day.pdf")

plt_predictive(40)
plt_predictive(100)
plt_predictive(200)


toplot = df[['doubling_time','beta', 'hosp_prop',
       'ICU_prop', 'vent_prop', 'hosp_LOS', 'ICU_LOS', 'vent_LOS', 'incubation_days' , 'recovery_days', 'logistic_k', 'logistic_x0',
       'logistic_L', 'nu', 'days_until_overacpacity', 'peak_demand', 'posterior']]
toplot.days_until_overacpacity[toplot.days_until_overacpacity == -9999] = np.nan

pspace = np.linspace(.001, .999, 1000)

fig, ax = plt.subplots(figsize=(8, 40), ncols=1, nrows=len(toplot.columns[:-3]))
for i in range(len(toplot.columns[:-3])):
    cname = toplot.columns[i]
    if params.loc[params.param == cname, 'distribution'].iloc[0] == 'gamma':    
        x = sps.gamma.ppf(pspace, params.loc[params.param == cname, 'p1'], 0, params.loc[params.param == cname, 'p2'])
        y = sps.gamma.pdf(x, params.loc[params.param == cname, 'p1'], 0, params.loc[params.param == cname, 'p2'])
    if params.loc[params.param == cname, 'distribution'].iloc[0] == 'beta':    
        x = sps.beta.ppf(pspace, params.loc[params.param == cname, 'p1'], params.loc[params.param == cname, 'p2'])
        y = sps.beta.pdf(x, params.loc[params.param == cname, 'p1'], params.loc[params.param == cname, 'p2'])
    ax[i].plot(x, y, label = "prior")
    ax[i].hist(toplot[cname], density = True, label = "posterior", bins=30)
    ax[i].set_xlabel(params.loc[params.param == cname, 'description'].iloc[0])
    ax[i].legend()
plt.tight_layout()
fig.savefig(f'{figdir}{hospital}_marginal_posteriors_v2.pdf')
