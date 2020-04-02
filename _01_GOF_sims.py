

import pandas as pd
import os
from _99_shared_functions import *
import multiprocessing as mp
import matplotlib.pyplot as plt

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

datadir = f'{os.getcwd()}/data/'
outdir = f'{os.getcwd()}/output/'

# import the census time series and set the zero day to be the first instance of zero
when_to_start = 7
census_ts = pd.read_csv(f"{datadir}census_ts.csv").iloc[when_to_start:,:]
census_ts.census *=.3
nobs = census_ts.shape[0]

# define vent capacity
vent_capacity = 183

seed = np.random.choice(10000000)

def wrapper(seed):
    try:
        # compute stats
        x = sensitivity_wrapper(seed = seed)
        out = x['parms']
        # get GOF
        out.update({"mse":np.mean((x['arr'][:nobs,5] - census_ts.census)**2)})
        out.update({"mse_last_week":np.mean((x['arr'][:nobs,5][-7:] - census_ts.census[-7:])**2)})
        # proj out -- days from today until vent capacity
        out.update({"days_until_overacpacity": int(np.apply_along_axis(lambda x: np.where((x - vent_capacity) > 0)[0][0] \
                                                                if max(x) > vent_capacity else -9999, axis=0,
                                                                arr=x['arr'][:,5]))})
        # get peak demands
        out.update({"peak_demand":np.max(x['arr'][:,5])})
        return out
    except Exception as e:
        return e


pool = mp.Pool(mp.cpu_count())
outdicts = pool.map(wrapper, range(10000))
pool.close()

outsims = pd.DataFrame(outdicts)
outsims.to_pickle(f"{outdir}sims_Apr2.pkl")
outsims = outsims.loc[outsims.mse_last_week**.5 < 20]
print(outsims.shape[0])

fig, ax = plt.subplots(figsize=(16, 10), ncols=2, nrows=2, sharex=False)
# histogram of days until overcapacity
plt.hist(outsims.days_until_overacpacity)
plt.show()
plt.cs
plt.hist(outsims.mse_last_week**.5)
plt.show()



