import re
import os

import numpy as np
import pandas as pd
import scipy.stats as sps

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000


def write_txt(str, path):
    text_file = open(path, "w")
    text_file.write(str)
    text_file.close()


# SIR simulation
def sir(y, alpha, beta, gamma, nu, N):
    S, E, I, R = y
    Sn = (-beta * (S / N) ** nu * I) + S
    En = (beta * (S / N) ** nu * I - alpha * E) + E
    In = (alpha * E - gamma * I) + I
    Rn = gamma * I + R

    scale = N / (Sn + En + In + Rn)
    return Sn * scale, En * scale, In * scale, Rn * scale


def reopenfn(day, reopen_day=60, reopen_speed=0.1, reopen_cap = .5):
    """Starting on `reopen_day`, reduce contact restrictions
    by `reopen_speed`*100%.
    """
    if day < reopen_day:
        return 1.0
    else:
        val = (1 - reopen_speed) ** (day - reopen_day)
        return val if val >= reopen_cap else reopen_cap

def reopen_wrapper(dfi, day, speed, cap):
    p_df = dfi.reset_index()   
    p_df.columns = ['param', 'val']
    ro = dict(param = ['reopen_day', 'reopen_speed', 'reopen_cap'],
              val = [day, speed, cap])
    p_df = pd.concat([p_df, pd.DataFrame(ro)])
    p_df
    SIR_ii = SIR_from_params(p_df)
    return SIR_ii['arr_stoch'][:,3]


def scale(arr, mu, sig):
    if len(arr.shape)==1:
        arr = np.expand_dims(arr, 0)
    arr = np.apply_along_axis(lambda x: x-mu, 1, arr)
    arr = np.apply_along_axis(lambda x: x/sig, 1, arr)
    return arr



# Run the SIR model forward in time
def sim_sir(
    S,
    E,
    I,
    R,
    alpha,
    beta,
    b0,
    beta_spline,
    beta_k,
    beta_spline_power,
    nobs,
    Xmu,
    Xsig,
    gamma,
    nu,
    n_days,
    logistic_L,
    logistic_k,
    logistic_x0,
    reopen_day = 8675309,
    reopen_speed = 0.0,
    reopen_cap = 1.0,
):
    N = S + E + I + R
    s, e, i, r = [S], [E], [I], [R]
    if len(beta_spline) > 0:
        knots = np.linspace(0, nobs-nobs/beta_k/2, beta_k)
    for day in range(n_days):
        y = S, E, I, R
        # evaluate splines
        if len(beta_spline) > 0:
            X = power_spline(day, knots, beta_spline_power, xtrim = nobs)
            # X = scale(X, Xmu, Xsig)
            #scale to prevent overflows and make the penalties comparable across bases
            XB = float(X@beta_spline)
            sd = logistic(L = 1, k=1, x0 = 0, x= b0 + XB)
        else:
            sd = logistic(logistic_L, logistic_k, logistic_x0, x=day)
        sd *= reopenfn(day, reopen_day, reopen_speed, reopen_cap)
        beta_t = beta * (1 - sd)
        S, E, I, R = sir(y, alpha, beta_t, gamma, nu, N)
        s.append(S)
        e.append(E)
        i.append(I)
        r.append(R)
    s, e, i, r = np.array(s), np.array(e), np.array(i), np.array(r)
    return s, e, i, r


# # compute X scale factor.  first need to compute who X matrix across all days
# nobs = 100
# n_days = 100
# beta_spline_power = 2
# beta_spline = np.random.uniform(size = len(knots))
# X = np.stack([power_spline(day, knots, beta_spline_power, xtrim = nobs) for day in range(n_days)])
# # need to be careful with this:  apply the scaling to the new X's when predicting



def power_spline(x, knots, n, xtrim):
    if x > xtrim: #trim the ends of the spline to prevent nonsense extrapolation
        x = xtrim + 1
    spl = x - np.array(knots)
    spl[spl<0] = 0
    spl = spl/(xtrim**n)#scaling -- xtrim is the max number of days, so the highest value that the spline could have
    return spl**n

'''
Plan:  
    beta_t = L/(1 + np.exp(XB))
'''

def logistic(L, k, x0, x):
    exp_term = np.exp(-k * (x - x0))
    # Catch overflow and return nan instead of 0.0
    if not np.isfinite(exp_term):
        return np.nan
    return L / (1 + exp_term)


def qdraw(qvec, p_df):
    """
    Function takes a vector of quantiles and returns marginals based on the parameters in the parameter data frame
    It returns a bunch of parameters for inputting into SIR
    It'll also return their probability under the prior
    """
    assert len(qvec) == p_df.shape[0]
    outdicts = []
    for i in range(len(qvec)):
        if p_df.distribution.iloc[i] == "constant":
            out = dict(param=p_df.param.iloc[i], val=p_df.base.iloc[i], prob=1)
        else:
            # Construct this differently for different distributoons
            if p_df.distribution.iloc[i] == "gamma":
                p = (qvec[i], p_df.p1.iloc[i], 0, p_df.p2.iloc[i])
            elif p_df.distribution.iloc[i] == "beta":
                p = (qvec[i], p_df.p1.iloc[i], p_df.p2.iloc[i])
            elif p_df.distribution.iloc[i] == "uniform":
                p = (qvec[i], p_df.p1.iloc[i], p_df.p1.iloc[i] + p_df.p2.iloc[i])
            elif p_df.distribution.iloc[i] == "norm":
                p = (qvec[i], p_df.p1.iloc[i], p_df.p2.iloc[i])
            out = dict(
                param=p_df.param.iloc[i],
                val=getattr(sps, p_df.distribution.iloc[i]).ppf(*p),
            )
            # does scipy not have a function to get the density from the quantile?
            p_pdf = (out["val"],) + p[1:]
            out.update({"prob": getattr(sps, p_df.distribution.iloc[i]).pdf(*p_pdf)})
        outdicts.append(out)
    return pd.DataFrame(outdicts)


def jumper(start, jump_sd):
    probit = sps.norm.ppf(start)
    probit += np.random.normal(size=len(probit), scale=jump_sd)
    newq = sps.norm.cdf(probit)
    return newq


def compute_census(projection_admits_series, mean_los):
    """Compute Census based on exponential LOS distribution."""
    census = [0]
    for a in projection_admits_series.values:
        c = float(a) + (1 - 1 / float(mean_los)) * census[-1]
        census.append(c)
    return np.array(census[1:])


def SIR_from_params(p_df):
    """
    This function takes the output from the qdraw function
    """
    n_hosp = int(p_df.val.loc[p_df.param == "n_hosp"])
    incubation_days = float(p_df.val.loc[p_df.param == "incubation_days"])
    hosp_prop = float(p_df.val.loc[p_df.param == "hosp_prop"])
    ICU_prop = float(p_df.val.loc[p_df.param == "ICU_prop"])
    vent_prop = float(p_df.val.loc[p_df.param == "vent_prop"])
    hosp_LOS = float(p_df.val.loc[p_df.param == "hosp_LOS"])
    ICU_LOS = float(p_df.val.loc[p_df.param == "ICU_LOS"])
    vent_LOS = float(p_df.val.loc[p_df.param == "vent_LOS"])
    recovery_days = float(p_df.val.loc[p_df.param == "recovery_days"])
    mkt_share = float(p_df.val.loc[p_df.param == "mkt_share"])
    region_pop = float(p_df.val.loc[p_df.param == "region_pop"])
    logistic_k = float(p_df.val.loc[p_df.param == "logistic_k"])
    logistic_L = float(p_df.val.loc[p_df.param == "logistic_L"])
    logistic_x0 = float(p_df.val.loc[p_df.param == "logistic_x0"])
    nu = float(p_df.val.loc[p_df.param == "nu"])
    beta = float(
        p_df.val.loc[p_df.param == "beta"]
    )  # get beta directly rather than via doubling time
    # assemble the coefficient vector for the splines
    beta_spline = np.array(p_df.val.loc[p_df.param.str.contains('beta_spline_coef')]) #this evaluates to an empty array if it's not in the params
    if len(beta_spline) > 0:
        b0 = float(p_df.val.loc[p_df.param == "b0"])
        beta_spline_power = np.array(p_df.val.loc[p_df.param == "beta_spline_power"])
        nobs = float(p_df.val.loc[p_df.param == "nobs"])
        beta_k = int(p_df.loc[p_df.param == "beta_spline_dimension", 'val'])
        Xmu = p_df.loc[p_df.param == "Xmu", 'val'].iloc[0]
        Xsig = p_df.loc[p_df.param == "Xsig", 'val'].iloc[0]
    else:
        beta_spline_power = None
        beta_k = None
        nobs = None
        b0 = None
        Xmu, Xsig = None, None
        
    reopen_day, reopen_speed, reopen_cap = 1000, 0.0, 1.0
    if "reopen_day" in p_df.param.values:
        reopen_day = int(p_df.val.loc[p_df.param == "reopen_day"])
    if "reopen_speed" in p_df.param.values:
        reopen_speed = float(p_df.val.loc[p_df.param == "reopen_speed"])
    if "reopen_cap" in p_df.param.values:
        reopen_cap = float(p_df.val.loc[p_df.param == "reopen_cap"])
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
    offset = expon.ppf(
        0.99, 1 / incubation_days
    )  # Enough time for 95% of exposed to become infected
    offset = int(offset)
    s, e, i, r = sim_sir(
        S=region_pop - total_infections,
        E=total_infections,
        I=0.0,  # n_infec / detection_prob,
        R=0.0,
        alpha=alpha,
        beta=beta,
        b0=b0,
        beta_spline = beta_spline,
        beta_k = beta_k,
        beta_spline_power = beta_spline_power,
        Xmu = Xmu,
        Xsig = Xsig,
        nobs = nobs,
        gamma=gamma,
        nu=nu,
        n_days=n_days + offset,
        logistic_L=logistic_L,
        logistic_k=logistic_k,
        logistic_x0=logistic_x0 + offset,
        reopen_day=reopen_day,
        reopen_speed=reopen_speed,
        reopen_cap=reopen_cap
    )

    arrs = {}
    for sim_type in ["mean", "stochastic"]:
        if sim_type == "mean":

            ds = np.diff(i) + np.diff(r)  # new infections is delta i plus delta r
            ds = np.array([0] + list(ds))
            ds = ds[offset:]

            hosp_raw = hosp_prop
            ICU_raw = hosp_raw * ICU_prop  # coef param
            vent_raw = ICU_raw * vent_prop  # coef param

            hosp = ds * hosp_raw * mkt_share
            icu = ds * ICU_raw * mkt_share
            vent = ds * vent_raw * mkt_share
        elif sim_type == "stochastic":
            # Sampling Stochastic Observation

            ds = np.diff(i) + np.diff(r)  # new infections is delta i plus delta r
            ds = np.array([0] + list(ds))

            #  Sample from expected new infections as
            #  a proportion of Exposed + Succeptible
            #  NOTE: This is still an *underaccounting* of stochastic
            #        process which would compound over time.
            #        This would require that the SEIR were truly stocastic.
            stocastic_dist = "binomial"
            if stocastic_dist == "binomial":
                #  Discrete individuals
                e_int = e.astype(int) + s.astype(int)
                prob_i = pd.Series(ds / e_int).fillna(0.0)
                prob_i = prob_i.apply(lambda x: min(x, 1.0))
                prob_i = prob_i.apply(lambda x: max(x, 0.0))
                ds = np.random.binomial(e_int, prob_i)
                ds = ds[offset:]

                #  Sample admissions as proportion of
                #  new infections.
                hosp = np.random.binomial(ds.astype(int), hosp_prop * mkt_share)
                icu = np.random.binomial(hosp, ICU_prop)
                vent = np.random.binomial(icu, vent_prop)
            elif stocastic_dist == "beta":
                #  Continuous fractions of individuals
                e_int = e + s
                prob_i = pd.Series(ds / e_int).fillna(0.0)
                prob_i = prob_i.apply(lambda x: min(x, 1.0))
                prob_i = prob_i.apply(lambda x: max(x, 0.0))
                ds = (
                    np.random.beta(prob_i * e_int + 1, (1 - prob_i) * e_int + 1) * e_int
                )
                ds = ds[offset:]

                #  Sample admissions as proportion of
                #  new infections.
                hosp = (
                    np.random.beta(
                        ds * hosp_prop * mkt_share + 1,
                        ds * (1 - hosp_prop * mkt_share) + 1,
                    )
                    * ds
                )
                icu = (
                    np.random.beta(hosp * ICU_prop + 1, hosp * (1 - ICU_prop) + 1)
                    * hosp
                )
                vent = (
                    np.random.beta(icu * vent_prop + 1, icu * (1 - vent_prop) + 1) * icu
                )

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
            census = compute_census(
                projection_admits[re.sub("_census", "_adm", k)], los
            )
            census_dict[k] = census
        proj = pd.concat([projection_admits, pd.DataFrame(census_dict)], axis=1)
        proj = proj.fillna(0)
        arrs[sim_type] = proj
    output = dict(
        days=np.asarray(proj.day),
        arr=np.asarray(arrs["mean"])[:, 1:],
        arr_stoch=np.asarray(arrs["stochastic"])[:, 1:],
        names=proj.columns.tolist()[1:],
        parms=p_df,
        s=s,
        e=e,
        i=i,
        r=r,
        offset=offset,
    )
    return output
