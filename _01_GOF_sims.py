

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

# import the census time series and set the zero day to be the first instance of zero
when_to_start = 0
census_ts = pd.read_csv(f"{datadir}census_ts.csv")#.iloc[when_to_start:,:]
census_ts.census *=.3 # this turns hosp into vent, crudely
nobs = census_ts.shape[0]

# define vent capacity
vent_capacity = 183


def eval_pos(pos):
    '''function takes quantiles of the priors and outputs a posterior and relevant stats'''
    draw = SIR_from_params(qdraw(pos))
    residuals = draw['arr'][:nobs,5][-7:] - census_ts.census[-7:] # 5 corresponds with vent census
    sigma2 = np.var(residuals)
    LL = np.sum(np.log((residuals**2)/(2*sigma2)))
    Lprior = np.log(draw['parms'].prob).sum()
    posterior = LL + Lprior
    out = dict(pos = pos,
               draw = draw,
               posterior = posterior)
    return(out)

# list of dictionaries, to put together later
outdicts = []
a,b = [], []
jump_sd = .1

# initial conditions
current_pos = eval_pos(np.repeat(.5, 13))

for ii in range(10000):
    try:
        proposed_pos = eval_pos(jumper(current_pos['pos'], .1))
        p_accept = np.exp(proposed_pos['posterior']-current_pos['posterior'])
        alpha = np.random.uniform(0,1)
        b.append(proposed_pos['posterior'])
        if alpha < p_accept:
            current_pos = proposed_pos

    except Exception as e:
        print(e)
    # append the relevant results
    current_pos['draw']['parms']
    out = {current_pos['draw']['parms'].param[i]:current_pos['draw']['parms'].val[i] for i in range(13)}
    out.update({"days_until_overacpacity": int(np.apply_along_axis(lambda x: np.where((x - vent_capacity) > 0)[0][0] \
                                                            if max(x) > vent_capacity else -9999, axis=0,
                                                            arr=current_pos['draw']['arr'][:,5]))})
    out.update({"peak_demand":np.max(current_pos['draw']['arr'][:,5])})
    outdicts.append(out)

plt.plot(b)

df = pd.DataFrame(outdicts)
df.head()

plt.plot(df.vent_LOS)

current_pos['draw']['parms']
