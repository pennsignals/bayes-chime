


import pandas as pd
import os
from _99_shared_functions import *
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy import stats as sps

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

datadir = f'{os.getcwd()}/data/'
outdir = f'{os.getcwd()}/output/'
figdir = f'{os.getcwd()}/figures/'

# import the census time series and set the zero day to be the first instance of zero
census_ts = pd.read_csv(f"{datadir}census_ts.csv")
# impute vent with the proportion of hosp.  this is a crude hack
census_ts.loc[census_ts.vent.isna(), 'vent'] = census_ts.hosp.loc[census_ts.vent.isna()]*np.mean(census_ts.vent/census_ts.hosp)

nobs = census_ts.shape[0]

# define vent capacity
vent_capacity = 183


df = pd.read_pickle(f'{outdir}chains.pkl')

# remove burnin
df = df.loc[(df.iter>1000) & (~df.chain.isin([1, 12]))]


# plot of logistic curves
def logistic(L, k, x0, x):
    return L/(1+np.exp(-k*(x-x0)))

qlist = []
for day in range(census_ts.shape[0]):
    ldist = logistic(df.logistic_L, 
                      df.logistic_k,
                      df.logistic_x0,
                      day)
    qlist.append(np.quantile(ldist, [.025, .5, .975]))

qmat = np.vstack(qlist)
# logistic SD plot
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
plt.xlabel(f'Days since march 14')
fig.savefig(f"{figdir}effective_soc_dist.pdf")


# residuals plot
meanres = [df.residuals_hosp.iloc[i].mean() for i in range(df.shape[0])]
df['meanres'] = meanres
plt.plot(df.groupby('iter').mean().meanres)



# plot of chains
plt.scatter(x = df.groupby(['chain']).mean()[['posterior']], y = df.groupby(['chain']).mean()[['doubling_time']])
plt.ylabel(f'Doubling time mean by chain')
plt.xlabel(f'Posterior mean by chain')

# predictive plot
arrs = np.stack([df.arr.iloc[i] for i in range(df.shape[0])])
arrq = np.quantile(arrs, axis = 0, q = [.025, .25, .5, .75, .975])

howfar = 200
fig, ax = plt.subplots(figsize=(16, 8), ncols=2, nrows=1)
# hosp
ax[0].plot(arrq[2,:howfar,3], label = 'posterior median')
ax[0].set_ylabel(f'Hospital census', fontsize=12, fontweight='bold')
ax[0].set_xlabel(f'Days since March 14', fontsize=12, fontweight='bold')
ax[0].fill_between(x = list(range(howfar)),
                   y1 = arrq[0,:howfar,3],
                   y2 = arrq[4,:howfar,3], 
                   label = '95% Credible Region',
                   alpha = .3,
                   lw = 2,
                   edgecolor = "k")
ax[0].fill_between(x = list(range(howfar)),
                   y1 = arrq[1,:howfar,3],
                   y2 = arrq[3,:howfar,3], 
                   label = '50% Credible Region',
                   alpha = .3,
                   lw = 2,
                   edgecolor = "k")
ax[0].plot(list(range(len(census_ts.hosp))), census_ts.hosp, 
           color = "red",
           label = "observed")
ax[0].axhline(y=711+310 + 320, color='k', ls='--', label = "hospital capacity")

ax[0].legend()



ax[1].plot(arrq[2,:howfar,5], label = 'posterior median')
ax[1].set_ylabel(f'Vent census', fontsize=12, fontweight='bold')
ax[1].set_xlabel(f'Days since March 14', fontsize=12, fontweight='bold')
ax[1].fill_between(x = list(range(howfar)),
                   y1 = arrq[0,:howfar,5],
                   y2 = arrq[4,:howfar,5], 
                   label = '95% Credible Region',
                   alpha = .3,
                   lw = 2,
                   edgecolor = "k")
ax[1].fill_between(x = list(range(howfar)),
                   y1 = arrq[1,:howfar,5],
                   y2 = arrq[3,:howfar,5], 
                   label = '50% Credible Region',
                   alpha = .3,
                   lw = 2,
                   edgecolor = "k")
ax[1].axhline(y=183, color='k', ls='--', label = "vent capacity")
ax[1].plot(list(range(len(census_ts.vent))), census_ts.vent, 
           color = "red",
           label = "observed")
ax[1].legend()
fig.savefig(f"{figdir}forecast_{howfar}_day.pdf")



plt.plot(arrq[1,:40,3])
plt.plot(arrq[0,:40,3])
plt.plot(arrq[2,:40,3])
plt.plot(list(range(len(census_ts.hosp))), census_ts.hosp)

plt.plot(arrq[1,:30,5])
plt.plot(arrq[0,:30,5])
plt.plot(arrq[2,:30,5])
plt.plot(list(range(len(census_ts.hosp))), census_ts.vent)
# plt.plot(b)

# residuals

var_i = 1
for var_i in range(params.shape[0]):
    p_space = np.linspace(0.01,0.99)
    ll = np.zeros(p_space.shape[0])
    for i, p_val in enumerate(p_space):
        sq  = np.array([0.5 for p in range(params.shape[0])])
        sq[var_i] = p_val
        try:
            eval_ = eval_pos(sq)
            ll[i] = eval_['posterior']
        except:
            ll[i] = np.nan
    fig, ax = plt.subplots(1,1, figsize=(8,1))
    ax.plot(p_space, ll)
    ax.set_title(params['description'].values[var_i])
    plt.show()

# df = pd.DataFrame(outdicts)

toplot = df[['doubling_time', 'hosp_prop',
       'ICU_prop', 'vent_prop', 'hosp_LOS', 'ICU_LOS', 'vent_LOS', 'incubation_days' , 'recovery_days', 'logistic_k', 'logistic_x0',
       'logistic_L', 'days_until_overacpacity', 'peak_demand', 'posterior']]
toplot.days_until_overacpacity[toplot.days_until_overacpacity == -9999] = np.nan

fig, ax = plt.subplots(figsize=(8, 80), ncols=1, nrows=toplot.shape[1])
for i in range(toplot.shape[1]):
    ax[i].hist(toplot.iloc[:,i])
    ax[i].set_title(toplot.columns[i], fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{figdir}marginal_posteriors.pdf')
plt.tight_layout()


params
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
fig.savefig(f'{figdir}marginal_posteriors_v2.pdf')
plt.tight_layout()

    
#         if p_df.distribution.iloc[i] == 'gamma':
#             p = (qvec[i],p_df.p1.iloc[i], 0, p_df.p2.iloc[i])
#         elif p_df.distribution.iloc[i] == 'beta':
#             p = (qvec[i],p_df.p1.iloc[i], p_df.p2.iloc[i])
#         elif p_df.distribution.iloc[i] == 'uniform':
#             p = (qvec[i], p_df.p1.iloc[i], p_df.p1.iloc[i]+ p_df.p2.iloc[i])

# plt.plot(sps.norm.ppf(pspace))


#            if p_df.distribution.iloc[i] == 'gamma':
#                 p = (qvec[i],p_df.p1.iloc[i], 0, p_df.p2.iloc[i])
#             elif p_df.distribution.iloc[i] == 'beta':
#                 p = (qvec[i],p_df.p1.iloc[i], p_df.p2.iloc[i])
#             elif p_df.distribution.iloc[i] == 'uniform':
#                 p = (qvec[i], p_df.p1.iloc[i], p_df.p1.iloc[i]+ p_df.p2.iloc[i])
#             out = dict(param = p_df.param.iloc[i],
#                        val = getattr(sps, p_df.distribution.iloc[i]).ppf(*p))

# import seaborn as sns
# grid = sns.PairGrid(data= toplot,
#                     vars = ['doubling_time', 'hosp_prop',
#         'ICU_prop', 'vent_prop', 'hosp_LOS', 'ICU_LOS', 'vent_LOS', 'recovery_days', 'logistic_k', 'logistic_x0',
#         'logistic_L', 'days_until_overacpacity', 'peak_demand', 'posterior'], size = 4)
# grid = grid.map_upper(plt.scatter, color = 'darkred')
# grid = grid.map_diag(plt.hist, bins = 10, color = 'darkred', 
#                       edgecolor = 'k')# Map a density plot to the lower triangle
# grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')


# fig, ax = plt.subplots(figsize=(16, 10), ncols=2, nrows=1)

# #Peak vent demand
# ax[0].plot(np.arange(1, 2.05, .05)*100, vent_res[:,2])
# ax[0].fill_between(x = np.arange(1, 2.05, .05)*100
#                     ,y1=vent_res[:,0]
#                ,y2 = vent_res[:,4]
#                ,alpha=.3
#             ,lw=2
#             ,edgecolor='k'
#                )
# ax[0].set_title(f'\nMortality by vent capacity increases', fontsize=18, fontweight='bold')
# ax[0].grid("on")
# ax[0].set_ylabel(f'Total mortality', fontsize=12, fontweight='bold')
# ax[0].set_xlabel(f'Percent of current vent capacity', fontsize=12, fontweight='bold')


# ax[1].plot(np.arange(0, .55, .05)*100, soc_res[:,2])
# ax[1].fill_between(x = np.arange(0, .55, .05)*100
#                     ,y1=soc_res[:,0]
#                ,y2 = soc_res[:,4]
#                ,alpha=.3
#             ,lw=2
#             ,edgecolor='k'
#                )
# ax[1].set_title(f'\nMortality by increasing social distancing', fontsize=18, fontweight='bold')
# ax[1].grid("on")
# ax[1].set_ylabel(f'Total mortality', fontsize=12, fontweight='bold')
# ax[1].set_xlabel(f'Percent decrease in social contact', fontsize=12, fontweight='bold')

# plt.tight_layout()
# fig.savefig(f'{figdir}mort_comparison_v1.pdf')
# plt.tight_layout()