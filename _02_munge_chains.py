


import pandas as pd
from os import getcwd, path
from _99_shared_functions import *
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy import stats as sps

import sys
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

# Get parameters from files stored during step 1, don't get from args (one argument, directory)
datadir = path.join(f'{getcwd()}', 'data')
outdir = path.join(f'{getcwd()}', 'output')
figdir = path.join(f'{getcwd()}', 'figures')

# import the census time series and set the zero day to be the first instance of zero
# Ditto step 1 - get from args etc...
for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}: {arg}")

hospital = sys.argv[1]
census_ts = pd.read_csv(path.join(f"{datadir}",f"{hospital}_ts.csv"))
first_day = census_ts['date'].values[0]
params = pd.read_csv(path.join(f"{datadir}",f"{hospital}_parameters.csv"))
# impute vent with the proportion of hosp.  this is a crude hack
census_ts.loc[census_ts.vent.isna(), 'vent'] = census_ts.hosp.loc[census_ts.vent.isna()]*np.mean(census_ts.vent/census_ts.hosp)

# This needs to be configuable based on the time period specificed 
nobs = census_ts.shape[0]

# define capacity
vent_capacity = float(params.base.loc[params.param == 'vent_capacity'])
hosp_capacity = float(params.base.loc[params.param == 'hosp_capacity'])

# Chains
df = pd.read_pickle(path.join(f'{outdir}',f'{hospital}_chains.pkl'))

# remove burn-in
# Make 1000 configurable
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
plt.ylim(0,1)
fig.savefig(path.join(f"{figdir}",f"{hospital}_effective_soc_dist.pdf"))



# plot of chains
def plt_predictive(howfar=200):
    # predictive plot
    arrs = np.stack([df.arr.iloc[i] for i in range(df.shape[0])])
    arrq = np.quantile(arrs, axis = 0, q = [.05, .25, .5, .75, .95])

    dates = pd.date_range(f'{first_day}',
        periods=howfar, freq='d')
    fig, ax = plt.subplots(figsize=(16, 10), ncols=2, nrows=2, sharex=True)
    # hosp
    axx = ax[0,0]
    axx.plot_date(dates, arrq[2,:howfar,3], '-', label = 'posterior median')
    axx.set_ylabel(f'COVID-19 Hospital census', fontsize=12, fontweight='bold')
    #axx.set_xlabel(f'Days since {first_day}', fontsize=12, fontweight='bold')
    axx.fill_between(x = dates,
                       y1 = arrq[0,:howfar,3],
                       y2 = arrq[4,:howfar,3], 
                       label = '90% Credible Region',
                       alpha = .1,
                       lw = 2,
                       edgecolor = "k")
    axx.fill_between(x = dates,
                       y1 = arrq[1,:howfar,3],
                       y2 = arrq[3,:howfar,3], 
                       label = '50% Credible Region',
                       alpha = .1,
                       lw = 2,
                       edgecolor = "k")
    axx.plot_date(dates[:census_ts.hosp.shape[0]], census_ts.hosp, '-',
               color = "red",
               label = "observed")
    axx.axhline(y=hosp_capacity, color='k', ls='--', label = "hospital capacity")
    axx.legend()
    axx.grid(True)

    axx = ax[0,1]
    axx.plot_date(dates, arrq[2,:howfar,5], '-', label = 'posterior median')
    axx.set_ylabel(f'COVID-19 Vent census', fontsize=12, fontweight='bold')
    #axx.set_xlabel(f'Days since {first_day}', fontsize=12, fontweight='bold')
    axx.fill_between(x = dates,
                       y1 = arrq[0,:howfar,5],
                       y2 = arrq[4,:howfar,5], 
                       label = '90% Credible Region',
                       alpha = .1,
                       lw = 2,
                       edgecolor = "k",)
    axx.fill_between(x = dates,
                       y1 = arrq[1,:howfar,5],
                       y2 = arrq[3,:howfar,5], 
                       label = '50% Credible Region',
                       alpha = .1,
                       lw = 2,
                       edgecolor = "k")
    axx.axhline(y=vent_capacity, color='k', ls='--', label = "vent capacity")
    axx.plot_date(dates[:census_ts.vent.shape[0]], census_ts.vent, '-',
               color = "red",
               label = "observed")
    axx.legend()
    axx.grid(True)

    # Admits
    axx = ax[1,0]
    axx.plot_date(dates,arrq[2,:howfar,0], '-', label = 'posterior median')
    axx.set_ylabel(f'COVID-19 Hospital Admits', fontsize=12, fontweight='bold')
    #axx.set_xlabel(f'Days since {first_day}', fontsize=12, fontweight='bold')
    axx.fill_between(x = dates,
                       y1 = arrq[0,:howfar,0],
                       y2 = arrq[4,:howfar,0], 
                       label = '90% Credible Region',
                       alpha = .1,
                       lw = 2,
                       edgecolor = "k")
    axx.fill_between(x = dates,
                       y1 = arrq[1,:howfar,0],
                       y2 = arrq[3,:howfar,0], 
                       label = '50% Credible Region',
                       alpha = .1,
                       lw = 2,
                       edgecolor = "k")
    axx.legend()
    axx.grid(True)

    axx = ax[1,1]
    axx.plot_date(dates, arrq[2,:howfar,2], '-', label = 'posterior median')
    axx.set_ylabel(f'COVID-19 Vent Admits', fontsize=12, fontweight='bold')
    #axx.set_xlabel(f'Days since {first_day}', fontsize=12, fontweight='bold')
    axx.fill_between(x = dates,
                       y1 = arrq[0,:howfar,2],
                       y2 = arrq[4,:howfar,2], 
                       label = '90% Credible Region',
                       alpha = .1,
                       lw = 2,
                       edgecolor = "k",)
    axx.fill_between(x = dates,
                       y1 = arrq[1,:howfar,2],
                       y2 = arrq[3,:howfar,2], 
                       label = '50% Credible Region',
                       alpha = .1,
                       lw = 2,
                       edgecolor = "k")
    axx.legend()
    axx.grid(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path.join(f"{figdir}",f"{hospital}_forecast_{howfar}_day.pdf"))

plt_predictive(40)
plt_predictive(100)
plt_predictive(200)


def mk_projection_tables():
    # predictive plot
    arrs = np.stack([df.arr.iloc[i] for i in range(df.shape[0])])
    arrq = np.quantile(arrs, axis = 0, q = [.05, .25, .5, .75, .95])
    column_postfix = ['5%', '25%', 'Median', '75%', '%95']

    summary_df_hsp = pd.DataFrame(data=arrq[:,:,3].T,
        columns=[f'Hospitalized Census {pf}' for pf in column_postfix])
    summary_df_vent = pd.DataFrame(data=arrq[:,:,5].T,
        columns=[f'Vent Census {pf}' for pf in column_postfix])


    summary_df_hsp_admits = pd.DataFrame(data=arrq[:,:,0].T.astype(int),
        columns=[f'Hospitalized Admits {pf}' for pf in column_postfix])
    summary_df_vent_admits = pd.DataFrame(data=arrq[:,:,2].T.astype(int),
        columns=[f'Vent Admits {pf}' for pf in column_postfix])

    date_df = pd.DataFrame(data=pd.date_range(f'{first_day}',
        periods=summary_df_hsp.shape[0], freq='d'),
        columns = ['date'])

    summary_df = pd.concat([date_df,
      summary_df_hsp,
      summary_df_vent,
      summary_df_hsp_admits,
      summary_df_vent_admits], 1)
    summary_df.to_csv(path.join(f"{outdir}",f"{hospital}_forecast.csv"), index=False)

mk_projection_tables()


toplot = df[['beta',
             'hosp_prop',
             'ICU_prop',
             'vent_prop',
             'hosp_LOS',
             'ICU_LOS',
             'vent_LOS',
             'incubation_days',
             'recovery_days',
             'logistic_k',
             'logistic_x0',
             'logistic_L']]
#toplot.days_until_overacpacity[toplot.days_until_overacpacity == -9999] = np.nan

pspace = np.linspace(.001, .999, 1000)

fig, ax = plt.subplots(figsize=(8, 40), ncols=1, nrows=len(toplot.columns))
for i in range(len(toplot.columns)):
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
fig.savefig(path.join(f'{figdir}',f'{hospital}_marginal_posteriors_v2.pdf'))
