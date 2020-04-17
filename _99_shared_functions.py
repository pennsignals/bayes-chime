
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import copy
import scipy.stats as sps

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

# define relative paths
assert os.getcwd().split("/")[-1] == "chime_sims"
datadir = f"{os.getcwd()}/data/"
outdir = f"{os.getcwd()}/output/"
figdir = f"{os.getcwd()}/figures/"


def write_txt(str, path):
    text_file = open(path, "w")
    text_file.write(str)
    text_file.close()


# SIR simulation
def sir(y, alpha, beta, gamma, nu, N):
    S, E, I, R = y
    Sn = (-beta * (S / N)**nu * I ) + S
    En = (beta * (S / N)**nu * I - alpha * E) + E
    In = (alpha * E - gamma * I) + I
    Rn = gamma * I + R

    scale = N / (Sn + En + In + Rn)
    return Sn * scale, En * scale, In * scale, Rn * scale


def reopenfn(day, reopen_day=60, reopen_speed=0.1):
    """Starting on `reopen_day`, reduce contact restrictions
    by `reopen_speed`*100%.
    """
    if day < reopen_day:
        return 1.0
    else:
        return (1-reopen_speed)**(day-reopen_day)


# Run the SIR model forward in time
def sim_sir(S, E, I, R, alpha, beta, gamma, nu, n_days, 
    logistic_L, logistic_k, logistic_x0,
    reopen_day=1000, reopen_speed=0.0):
    N = S + E + I + R
    s, e, i, r = [S], [E], [I], [R]
    for day in range(n_days):
        y = S, E, I, R
        # evaluate logistic
        sd = logistic(logistic_L, logistic_k, logistic_x0, x = day)
        sd *= reopenfn(day, reopen_day, reopen_speed)
        beta_t = beta*(1-sd)
        S, E, I, R = sir(y, alpha, beta_t, gamma, nu, N)
        s.append(S)
        e.append(E)
        i.append(I)
        r.append(R)
    s, e, i, r = np.array(s), np.array(e), np.array(i), np.array(r)
    return s, e, i, r

def logistic(L, k, x0, x):
    return L/(1+np.exp(-k*(x-x0)))

# y = 1-logistic(.5, .5, 10, np.arange(0, 30))
# plt.plot(np.arange(0, 30), y)


def qdraw(qvec, p_df):
    '''
    Function takes a vector of quantiles and returns marginals based on the parameters in the parameter data frame
    It returns a bunch of parameters for inputting into SIR
    It'll also return their probability under the prior
    '''
    assert len(qvec) == p_df.shape[0]
    outdicts = []
    for i in range(len(qvec)):
        if p_df.distribution.iloc[i] == 'constant':
            out = dict(param = p_df.param.iloc[i],
                       val = p_df.base.iloc[i],
                       prob = 1)
        else:
            # Construct this differently for different distributoons
            if p_df.distribution.iloc[i] == 'gamma':
                p = (qvec[i],p_df.p1.iloc[i], 0, p_df.p2.iloc[i])
            elif p_df.distribution.iloc[i] == 'beta':
                p = (qvec[i],p_df.p1.iloc[i], p_df.p2.iloc[i])
            elif p_df.distribution.iloc[i] == 'uniform':
                p = (qvec[i], p_df.p1.iloc[i], p_df.p1.iloc[i]+ p_df.p2.iloc[i])
            out = dict(param = p_df.param.iloc[i],
                       val = getattr(sps, p_df.distribution.iloc[i]).ppf(*p))
            # does scipy not have a function to get the density from the quantile?
            p_pdf = (out['val'],) + p[1:]
            out.update({"prob": getattr(sps, p_df.distribution.iloc[i]).pdf(*p_pdf)})
        outdicts.append(out)
    return pd.DataFrame(outdicts)

def jumper(start, jump_sd):
    probit = sps.norm.ppf(start)
    probit += np.random.normal(size=len(probit), scale=jump_sd)
    newq = sps.norm.cdf(probit)
    return newq


def SIR_from_params(p_df):
    '''
    This function takes the output from the qdraw function
    '''
    #
    n_hosp = int(p_df.val.loc[p_df.param == 'n_hosp'])
    incubation_days = float(p_df.val.loc[p_df.param == 'incubation_days'])
    hosp_prop = float(p_df.val.loc[p_df.param == 'hosp_prop'])
    ICU_prop = float(p_df.val.loc[p_df.param == 'ICU_prop'])
    vent_prop = float(p_df.val.loc[p_df.param == 'vent_prop'])
    hosp_LOS = float(p_df.val.loc[p_df.param == 'hosp_LOS'])
    ICU_LOS = float(p_df.val.loc[p_df.param == 'ICU_LOS'])
    vent_LOS = float(p_df.val.loc[p_df.param == 'vent_LOS'])
    recovery_days = float(p_df.val.loc[p_df.param == 'recovery_days'])
    mkt_share = float(p_df.val.loc[p_df.param == 'mkt_share'])
    region_pop = float(p_df.val.loc[p_df.param == 'region_pop'])
    logistic_k = float(p_df.val.loc[p_df.param == 'logistic_k'])
    logistic_L = float(p_df.val.loc[p_df.param == 'logistic_L'])
    logistic_x0 = float(p_df.val.loc[p_df.param == 'logistic_x0'])
    beta = float(p_df.val.loc[p_df.param == 'beta'])  # get beta directly rather than via doubling time
    nu = float(p_df.val.loc[p_df.param == 'nu']) + 1.0

    reopen_day, reopen_speed = 1000, 0.0
    if 'reopen_day' in p_df.param.values:
        reopen_day = int(p_df.val.loc[p_df.param == 'reopen_day'])
    if 'reopen_speed' in p_df.param.values:
        reopen_speed = float(p_df.val.loc[p_df.param == 'reopen_speed'])
    #
    alpha = 1 / incubation_days
    gamma = 1 / recovery_days
    total_infections = n_hosp / mkt_share / hosp_prop

    n_days = 200

    # Offset by the incubation period to start the sim
    # that many days before the first hospitalization
    # Estimate the number Exposed from the number hospitalized
    # on the first day of non-zero covid hospitalizations.
    from scipy.stats import expon
    # Since incubation_days is exponential in SEIR, we start
    # the time `offset` days before the first hospitalization
    # We determine offset by allowing enough time for the majority
    # of the initial exposures to become infected.
    offset = expon.ppf(0.99, 1 / incubation_days)  # Enough time for 95% of exposed to become infected
    offset = int(offset)
    #
    s, e, i, r = sim_sir(S=region_pop - total_infections,
                      E=total_infections,
                      I=0.0,#n_infec / detection_prob,
                      R=0.0,
                      alpha=alpha,
                      beta=beta,
                      gamma=gamma,
                      nu=nu,
                      n_days=n_days + offset,
                      logistic_L = logistic_L,
                      logistic_k = logistic_k,
                      logistic_x0 = logistic_x0 + offset,
                      reopen_day = reopen_day,
                      reopen_speed = reopen_speed)

    ds = np.diff(i) + np.diff(r)  # new infections is delta i plus delta r
    ds = np.array([0] + list(ds))
    ds = ds[offset:]

    arrs = {}
    for sim_type in ['mean', 'stochastic']:
        if sim_type == 'mean':
            hosp_raw = hosp_prop
            ICU_raw = hosp_raw * ICU_prop  # coef param
            vent_raw = ICU_raw * vent_prop  # coef param

            hosp = ds * hosp_raw * mkt_share
            icu = ds * ICU_raw * mkt_share
            vent = ds * vent_raw * mkt_share
        elif sim_type == 'stochastic':
            # Sampling Stochastic Observation
            hosp = np.random.binomial(ds.astype(int), hosp_prop * mkt_share)
            icu = np.random.binomial(hosp, ICU_prop)
            vent = np.random.binomial(icu, vent_prop)

        # make a data frame with all the stats for plotting
        days = np.array(range(0, n_days + 1))
        data_list = [days, hosp, icu, vent]
        data_dict = dict(zip(["day", "hosp_adm", "icu_adm", "vent_adm"], data_list))
        projection = pd.DataFrame.from_dict(data_dict)
        projection_admits = projection
        projection_admits["day"] = range(projection_admits.shape[0])
        # census df
        hosp_LOS_raw = hosp_LOS
        ICU_LOS_raw = ICU_LOS
        vent_LOS_raw = vent_LOS

        los_dict = {
            "hosp_census": hosp_LOS_raw,
            "icu_census": ICU_LOS_raw,
            "vent_census": vent_LOS_raw,
        }
        census_dict = {}
        for k, los in los_dict.items():
            census = (
                    projection_admits.cumsum().iloc[:-int(los), :]
                    - projection_admits.cumsum().shift(int(los)).fillna(0)
            ).apply(np.ceil)
            census_dict[k] = census[re.sub("_census", "_adm", k)]
        proj = pd.concat([projection_admits, pd.DataFrame(census_dict)], axis=1)
        proj = proj.fillna(0)
        arrs[sim_type] = proj
    #
    output = dict(days=np.asarray(proj.day),
                  arr=np.asarray(arrs['mean'])[:, 1:],
                  arr_stoch=np.asarray(arrs['stochastic'])[:, 1:],
                  names=proj.columns.tolist()[1:],
                  parms=p_df,
                  s=s,
                  e=e,
                  i=i,
                  r=r,
                  offset=offset)
    return output
