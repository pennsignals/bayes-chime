


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
when_to_start = 1
census_ts = pd.read_csv(f"{datadir}census_ts.csv")#.iloc[when_to_start:,:]
# impute vent with the proportion of hosp.  this is a crude hack
census_ts.loc[census_ts.vent.isna(), 'vent'] = census_ts.hosp.loc[census_ts.vent.isna()]*np.mean(census_ts.vent/census_ts.hosp)

nobs = census_ts.shape[0]

# define vent capacity
vent_capacity = 183


df = pd.read_pickle(f'{outdir}chains.pkl')

# remove burnin
df = df.loc[df.iter>1000]

# plot of logistic curves
def logistic(L, k, x0, x):
    return L/(1+np.exp(-k*(x-x0)))

qlist = []
for day in range(census_ts.shape[0]):
    ldist = logistic(df.logistic_L, 
                      df.logistic_k,
                      df.logistic_x0,
                      day)
    qlist.append(np.quantile(ldist, [.25, .5, .75]))

qmat = np.vstack(qlist)




fig = plt.plot(figsize=(6, 8))

plt.plot(list(range(census_ts.shape[0])), 1-qmat[:,1])
plt.fill_between(x = list(range(census_ts.shape[0]))
                 ,y1=1-qmat[:,0]
                 ,y2 = 1-qmat[:,2]
                 ,alpha=.3
                 ,lw=2
                 ,edgecolor='k'
               )
plt.ylabel(f'Relative social contact')
plt.xlabel(f'Days since march 6')



# plt.plot(b)

# df = pd.DataFrame(outdicts)

toplot = df[['doubling_time', 'hosp_prop',
       'ICU_prop', 'vent_prop', 'hosp_LOS', 'ICU_LOS', 'vent_LOS', 'recovery_days', 'logistic_k', 'logistic_x0',
       'logistic_L', 'days_until_overacpacity', 'peak_demand', 'posterior']]
toplot.days_until_overacpacity[toplot.days_until_overacpacity == -9999] = np.nan

fig, ax = plt.subplots(figsize=(8, 80), ncols=1, nrows=toplot.shape[1])
for i in range(toplot.shape[1]):
    ax[i].hist(toplot.iloc[:,i])
    ax[i].set_title(toplot.columns[i], fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{figdir}foo.pdf')
plt.tight_layout()



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