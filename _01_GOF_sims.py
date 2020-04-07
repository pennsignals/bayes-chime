

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

def eval_pos(pos):
    '''function takes quantiles of the priors and outputs a posterior and relevant stats'''
    draw = SIR_from_params(qdraw(pos))
    # loss for vent
    residuals_vent = draw['arr'][:nobs,5] - census_ts.vent # 5 corresponds with vent census
    if any(residuals_vent == 0):
        residuals_vent[residuals_vent == 0] = .01
    sigma2 = np.var(residuals_vent)
    LLv = loglik(residuals_vent)#np.sum(-np.log((residuals_vent**2)/(2*sigma2)))
    Lpriorv = np.log(draw['parms'].prob).sum()
    posterior_vent = LLv + Lpriorv
    # loss for hosp
    residuals_hosp = draw['arr'][:nobs,3] - census_ts.hosp # 5 corresponds with vent census
    if any(residuals_hosp == 0):
        residuals_hosp[residuals_hosp == 0] = .01
    sigma2 = np.var(residuals_hosp)
    LLh = loglik(residuals_hosp)#np.sum(-np.log((residuals_hosp**2)/(2*sigma2)))    
    Lpriorh = np.log(draw['parms'].prob).sum()
    posterior_hosp = LLh + Lpriorh
    # average them
    posterior = np.mean([posterior_hosp, posterior_vent])    
    out = dict(pos = pos,
               draw = draw,
               posterior = posterior,
               residuals_vent = residuals_vent,
               residuals_hosp = residuals_hosp)
    return(out)


def loglik(r):
    return -len(r)/2*(np.log(2*np.pi*np.var(r))) - 1/(2*np.pi*np.var(r))*np.sum(r**2)

# specifying the standard deviation of the nump, in gaussian quantile space per the jumper function
jump_sd = .1
seed = 5

def chain(seed):
    np.random.seed(seed)
    current_pos = eval_pos(np.random.uniform(size = params.shape[0]))
    outdicts = []
    n_iters = 5000
    U = np.random.uniform(0,1,n_iters)
    for ii in range(n_iters):
        try:
            proposed_pos = eval_pos(jumper(current_pos['pos'], .1))
            p_accept = np.exp(proposed_pos['posterior']-current_pos['posterior'])

            if U[ii] < p_accept:
                current_pos = proposed_pos

        except Exception as e:
            print(e)
        # append the relevant results
        out = {current_pos['draw']['parms'].param[i]:current_pos['draw']['parms'].val[i] for i in range(params.shape[0])}
        out.update({"days_until_overacpacity": int(np.apply_along_axis(lambda x: np.where((x - vent_capacity) > 0)[0][0] \
                                                                if max(x) > vent_capacity else -9999, axis=0,
                                                                arr=current_pos['draw']['arr'][:,5]))})
        out.update({"peak_demand":np.max(current_pos['draw']['arr'][:,5])})
        out.update({"arr": current_pos['draw']['arr']})
        out.update({"iter":ii})
        out.update({"chain":seed})
        out.update({'posterior':proposed_pos['posterior']})
        out.update({'residuals_hosp':proposed_pos['residuals_hosp']})
        out.update({'residuals_vent':proposed_pos['residuals_vent']})
        outdicts.append(out)
        if (ii % 1000) == 0:
            print('chain', seed, 'iter', ii)
    return pd.DataFrame(outdicts)


pool = mp.Pool(mp.cpu_count())
chains = pool.map(chain, list(range(16)))
pool.close()

df = pd.concat(chains)
df.to_pickle(f'{outdir}chains.pkl')

# # remove burnin
# df = df.loc[df.iter>1000]

# # plot of logistic curves
# def logistic(L, k, x0, x):
#     return L/(1+np.exp(-k*(x-x0)))

# qlist = []
# for day in range(census_ts.shape[0]):
#     ldist = logistic(df.logistic_L, 
#                      df.logistic_k,
#                      df.logistic_x0,
#                      day)
#     qlist.append(np.quantile(ldist, [.025, .5, .975]))

# qmat = np.vstack(qlist)

# plt.plot(1-qmat[:,1])
# plt.plot(1-qmat[:,0])
# plt.plot(1-qmat[:,2])

# plt.plot(b)

# df = pd.DataFrame(outdicts)

# fig, ax = plt.subplots(figsize=(16, 40), ncols=1, nrows=17)
# for i in range(17):
#     ax[i].plot(df.iloc[:,i])
#     ax[i].set_title(df.columns[i], fontsize=12, fontweight='bold')
# plt.tight_layout()
# fig.savefig(f'{figdir}foo.pdf')
# plt.tight_layout()


