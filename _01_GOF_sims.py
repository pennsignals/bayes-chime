

import pandas as pd
import os
from _99_shared_functions import *
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy import stats as sps
from utils import *
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
n_chains = int(sys.argv[2])
n_iters = int(sys.argv[3])
penalty_factor = float(sys.argv[4])

census_ts = pd.read_csv(f"{datadir}{hospital}_ts.csv")
# import parameters
params = pd.read_csv(f"{datadir}{hospital}_parameters.csv")
# impute vent with the proportion of hosp.  this is a crude hack
census_ts.loc[census_ts.vent.isna(), 'vent'] = census_ts.hosp.loc[census_ts.vent.isna()]*np.mean(census_ts.vent/census_ts.hosp)

nobs = census_ts.shape[0]

# define vent capacity
vent_capacity = 183


def loglik(r):
    return -len(r)/2*(np.log(2*np.pi*np.var(r))) - 1/(2*np.pi*np.var(r))*np.sum(r**2)


def do_shrinkage(pos, shrinkage):
    densities = sps.beta.pdf(pos, a = shrinkage[0], b = shrinkage[1])
    regularization_penalty = -np.sum(np.log(densities))
    return regularization_penalty


def eval_pos(pos, shrinkage = None, holdout = 0):
    '''function takes quantiles of the priors and outputs a posterior and relevant stats'''
    draw = SIR_from_params(qdraw(pos, params))
    if holdout >0:
        train = census_ts[:-holdout]
        test = census_ts[-holdout:]
    else:
        train = census_ts
    # loss for vent
    LL = 0
    residuals_vent = None
    if train.vent.sum() > 0:
        residuals_vent = draw['arr'][:(nobs-holdout),5] - train.vent # 5 corresponds with vent census
        if any(residuals_vent == 0):
            residuals_vent[residuals_vent == 0] = .01
        sigma2 = np.var(residuals_vent)
        LL += loglik(residuals_vent)

    # loss for hosp
    residuals_hosp = draw['arr'][:(nobs-holdout),3] - train.hosp # 5 corresponds with vent census
    if any(residuals_hosp == 0):
        residuals_hosp[residuals_hosp == 0] = .01
    sigma2 = np.var(residuals_hosp)
    LL += loglik(residuals_hosp)

    Lprior = np.log(draw['parms'].prob).sum()
    posterior = LL + Lprior
    # shrinkage -- the regarization parameter reaches its max value at the median of each prior.
    # the penalty gets subtracted off of the posterior
    if shrinkage is not None:
        assert (str(type(shrinkage).__name__) == "ndarray") & (len(shrinkage) == 2)
        posterior -= do_shrinkage(pos, shrinkage)

    out = dict(pos = pos,
               draw = draw,
               posterior = posterior,
               residuals_vent = residuals_vent,
               residuals_hosp = residuals_hosp)
    if holdout > 0:
        res_te_vent = draw['arr'][(nobs-holdout):nobs,5] - test.vent
        res_te_hosp = draw['arr'][(nobs-holdout):nobs,3] - test.hosp
        test_loss = (np.mean(res_te_hosp**2) + np.mean(res_te_vent**2))/2
        out.update({"test_loss":test_loss})
    return(out)


# specifying the standard deviation of the jump, in gaussian quantile space per the jumper function
jump_sd = .1
seed = 5

def chain(seed, shrinkage = None, holdout = 0):
    np.random.seed(seed)
    if shrinkage is not None:
        assert (shrinkage < 1) and (shrinkage >= .05)
        sq1 = shrinkage/2
        sq2 = 1- shrinkage/2
        shrinkage = beta_from_q(sq1, sq2)
    current_pos = eval_pos(np.random.uniform(size = params.shape[0]), shrinkage = shrinkage, holdout = holdout)
    outdicts = []
    U = np.random.uniform(0, 1, n_iters)
    for ii in range(n_iters):
        try:
            proposed_pos = eval_pos(jumper(current_pos['pos'], .1), shrinkage = shrinkage, holdout = holdout)
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
        out.update({'offset': current_pos['draw']['offset']})
        if holdout > 0:
            out.update({'test_loss': current_pos['test_loss']})
        outdicts.append(out)
        if shrinkage is None:        
            if (ii % 1000) == 0:
                print('chain', seed, 'iter', ii)
    return pd.DataFrame(outdicts)




def loop_over_shrinkage(seed, holdout=7, shrvec = np.linspace(.05, .95, 10)):
    test_loss = []
    for shr in shrvec:
        chain_out = chain(seed, shr, holdout)
        test_loss.append(chain_out['test_loss'])
    return test_loss



if penalty_factor<0:
    pool = mp.Pool(mp.cpu_count())
    shrinkage_chains = pool.map(loop_over_shrinkage, list(range(n_chains)))
    pool.close()
    # put together the mp results
    master_chain = shrinkage_chains[0]
    for i in range(1, len(shrinkage_chains)):
        for j in range(len(master_chain)):
            master_chain[j] += shrinkage_chains[i][j]
    
    penalty = np.linspace(.05, .95, 10)
    mean_test_loss = [np.mean(i[1000:]) for i in master_chain]
    
    fig = plt.figure()
    plt.plot(penalty, mean_test_loss)
    plt.fill_between(x = penalty,
                     y1 = [float(np.quantile(i[1000:], [.025])) for i in master_chain],
                     y2 = [float(np.quantile(i[1000:], [.975])) for i in master_chain],
                     alpha=.3,
                     lw=2,
                     edgecolor='k')
    plt.xlabel('penalty factor')
    plt.ylabel('test MSE')
    fig.savefig(f"{figdir}{hospital}_shrinkage_grid_GOF.pdf")
    
    # identify the best penalty
    best_penalty = penalty[np.argmin(mean_test_loss)]
elif penalty_factor < 1:
    best_penalty = penalty_factor
    
tuples_for_starmap = [(i, best_penalty, 0) for i in range(n_chains)]

# get the final answer based on the best penalty
pool = mp.Pool(mp.cpu_count())
chains = pool.starmap(chain, tuples_for_starmap)
pool.close()

df = pd.concat(chains)
df.to_pickle(f'{outdir}{hospital}_chains.pkl')


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


