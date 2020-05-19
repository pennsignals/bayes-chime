
from copy import deepcopy
from datetime import datetime
from os import getcwd, path, makedirs
from string import ascii_letters, digits
import json
import multiprocessing as mp

from configargparse import ArgParser
from git import Repo
from scipy import stats as sps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


from _99_shared_functions import SIR_from_params, qdraw, jumper, power_spline,\
    reopen_wrapper

from _02_munge_chains import SD_plot, mk_projection_tables, plt_predictive, \
    plt_pairplot_posteriors, SEIR_plot, Rt_plot
from utils import beta_from_q

LET_NUMS = pd.Series(list(ascii_letters) + list(digits))
# PARAMDIR = None
# CENSUS_TS = None
# PARAMS = None
# NOBS = None
# N_ITERS = None

def get_dir_name(options):
    now = datetime.now()
    dir = now.strftime("%Y_%m_%d_%H_%M_%S")
    if options.prefix:
        dir = f"{dir}_{options.prefix}"
    if options.out:
        dir = f"{dir}_{options.out}"
    outdir = path.join(f"{getcwd()}", "output", dir)
    # In case we're running a few instances in a tight loop, generate a random
    # output directory
    if path.isdir(outdir):
        dir = f"{dir}_{''.join(LET_NUMS.sample(6, replace=True))}"
        outdir = path.join(f"{getcwd()}", "output", dir)
    makedirs(outdir)
    return outdir


def get_inputs(options):
    census_ts, params = None, None
    if options.prefix is not None:
        prefix = options.prefix
        datadir = path.join(f"{getcwd()}", "data")
        # import the census time series and set the zero day to be the first instance of zero
        census_ts = pd.read_csv(path.join(f"{datadir}", f"{prefix}_ts.csv"), encoding='latin-1')
        # impute vent with the proportion of hosp.  this is a crude hack
        census_ts.loc[census_ts.vent.isna(), "vent"] = census_ts.hosp.loc[
            census_ts.vent.isna()
        ] * np.mean(census_ts.vent / census_ts.hosp)
        # import parameters
        params = pd.read_csv(path.join(f"{datadir}", f"{prefix}_parameters.csv"), encoding = 'latin-1')
    if options.parameters is not None:
        params = pd.read_csv(options.parameters, encoding = 'latin-1')
    if options.ts is not None:
        census_ts = pd.read_csv(options.ts, encoding = 'latin-1')
        # impute vent with the proportion of hosp.  this is a crude hack
        census_ts.loc[census_ts.vent.isna(), "vent"] = census_ts.hosp.loc[
            census_ts.vent.isna()
        ] * np.mean(census_ts.vent / census_ts.hosp)
    return census_ts, params


def write_inputs(options, paramdir, census_ts, params):
    with open(path.join(paramdir, "args.json"), "w") as f:
        json.dump(options.__dict__, f)
    census_ts.to_csv(path.join(paramdir, "census_ts.csv"), index=False)
    params.to_csv(path.join(paramdir, "params.csv"), index=False)
    with open(path.join(paramdir, "git.sha"), "w") as f:
        f.write(Repo(search_parent_directories=True).head.object.hexsha)


def loglik(r):
    return -len(r) / 2 * (np.log(2 * np.pi * np.var(r))) - 1 / (
        2 * np.pi * np.var(r)
    ) * np.sum(r ** 2)


def do_shrinkage(pos, shrinkage, shrink_mask):
    densities = sps.beta.pdf(pos, a=shrinkage[0], b=shrinkage[1])
    densities *= shrink_mask
    regularization_penalty = -np.sum(np.log(densities))
    return regularization_penalty


def eval_pos(pos, params, obs, shrinkage, shrink_mask, holdout, 
             sample_obs, forecast_priors, ignore_vent):
    """function takes quantiles of the priors and outputs a posterior and relevant stats"""
    n_obs = obs.shape[0]
    nobs = n_obs-holdout
    draw = SIR_from_params(qdraw(pos, params))
    if sample_obs:
        ynoise_h = np.random.normal(scale=obs.hosp_rwstd)
        ynoise_h[0] = 0
        obs.hosp += ynoise_h
        ynoise_v = np.random.normal(scale=obs.vent_rwstd)
        ynoise_v[0] = 0
        obs.vent += ynoise_v
    if holdout > 0:
        train = obs[:-holdout]
        test = obs[-holdout:]
    else:
        train = obs
    # loss for vent
    LL = 0
    residuals_vent = None
    if train.vent.sum() > 0:
        residuals_vent = (
            draw["arr"][: (n_obs - holdout), 5] - train.vent.values[:nobs]
        )  # 5 corresponds with vent census
        if any(residuals_vent == 0):
            residuals_vent[residuals_vent == 0] = 0.01
        sigma2 = np.var(residuals_vent)
        if ignore_vent is False:
            LL += loglik(residuals_vent)

    # loss for hosp
    residuals_hosp = (
        draw["arr"][: (n_obs - holdout), 3] - train.hosp.values[:nobs]
    )  # 3 corresponds with hosp census
    if any(residuals_hosp == 0):
        residuals_hosp[residuals_hosp == 0] = 0.01
    sigma2 = np.var(residuals_hosp)
    LL += loglik(residuals_hosp)

    Lprior = np.log(draw["parms"].prob).sum()
    posterior = LL + Lprior
    # shrinkage -- the regarization parameter reaches its max value at the median of each prior.
    # the penalty gets subtracted off of the posterior
    if shrinkage is not None:
        assert (str(type(shrinkage).__name__) == "ndarray") & (len(shrinkage) == 2)
        posterior -= do_shrinkage(pos, shrinkage, shrink_mask)
    # forecast prior: compute the probability of the current forecast undet the specified prior
    # first compute the percent change in the forecast, one week out
    # then compute the probability of the change under the prior
    if forecast_priors['sig']>0:
        hosp_next_week = draw['arr'][n_obs+7,3]
        hosp_now = train.hosp.values[-1]
        hosp_pct_diff = (hosp_next_week/hosp_now-1) * 100
        hosp_forecast_prob = sps.norm.pdf(hosp_pct_diff, forecast_priors['mu'], forecast_priors['sig'])
        
        vent_next_week = draw['arr'][n_obs+7,5]
        vent_now = train.vent.values[-1]
        vent_pct_diff = (vent_next_week/vent_now-1) * 100
        vent_forecast_prob = sps.norm.pdf(vent_pct_diff, forecast_priors['mu'], forecast_priors['sig'])      

        forecast_prior_contrib = (hosp_forecast_prob * vent_forecast_prob)
        forecast_prior_contrib = np.log(forecast_prior_contrib) if forecast_prior_contrib >0 else -np.inf
        posterior += forecast_prior_contrib

    out = dict(
        pos=pos,
        draw=draw,
        posterior=posterior,
        residuals_vent=residuals_vent,
        residuals_hosp=residuals_hosp,
    )
    if holdout > 0:
        res_te_vent = draw["arr"][(n_obs - holdout) : n_obs, 5] - test.vent.values[:n_obs]
        res_te_hosp = draw["arr"][(n_obs - holdout) : n_obs, 3] - test.hosp.values[:n_obs]
        test_loss = (np.mean(res_te_hosp ** 2) + np.mean(res_te_vent ** 2)) / 2
        out.update({"test_loss": test_loss})
    return out


def chain(seed, params, obs, n_iters, shrinkage, holdout, 
          forecast_priors,
          sample_obs,
          ignore_vent):
    np.random.seed(seed)
    if shrinkage is not None:
        assert (shrinkage < 1) and (shrinkage >= 0.05)
        sq1 = shrinkage / 2
        sq2 = 1 - shrinkage / 2
        shrinkage = beta_from_q(sq1, sq2)
        shrink_mask= np.array([1 if "" in i else 0 for i in params.param])
    current_pos = eval_pos(
        pos = np.random.uniform(size=params.shape[0]),
        params = params,
        obs = obs, 
        shrinkage=shrinkage,
        shrink_mask = shrink_mask,
        holdout=holdout,
        sample_obs=sample_obs,
        forecast_priors = forecast_priors,
        ignore_vent = ignore_vent
    )
    outdicts = []
    U = np.random.uniform(0, 1, n_iters)
    posterior_history = []
    jump_sd = .2 # this is the starting value
    for ii in range(n_iters):
        try:
            proposed_pos = eval_pos(
                jumper(current_pos["pos"], jump_sd),
                params,
                obs,
                shrinkage=shrinkage,
                shrink_mask = shrink_mask,
                holdout=holdout,
                sample_obs=sample_obs,
                forecast_priors = forecast_priors,
                ignore_vent = ignore_vent
            )
            p_accept = np.exp(proposed_pos["posterior"] - current_pos["posterior"])
            if U[ii] < p_accept:
                current_pos = proposed_pos

        except Exception as e:
            print(e)
        # append the relevant results
        out = {
            current_pos["draw"]["parms"].param[i]: current_pos["draw"]["parms"].val[i]
            for i in range(params.shape[0])
        }
        # out.update({"arr": current_pos["draw"]["arr"]})
        out.update({"arr": current_pos["draw"]["arr_stoch"]})
        out.update({"iter": ii})
        out.update({"chain": seed})
        out.update({"posterior": proposed_pos["posterior"]})
        out.update({"offset": current_pos["draw"]["offset"]})
        out.update({"s": current_pos['draw']['s']})
        out.update({"e": current_pos['draw']['e']})
        out.update({"i": current_pos['draw']['i']})
        out.update({"r": current_pos['draw']['r']})
        if holdout > 0:
            out.update({"test_loss": current_pos["test_loss"]})
        outdicts.append(out)
        posterior_history.append(current_pos['posterior'])
        if (ii%100 == 0) and (ii>200):
            # diagnose:
            always_rejecting = len(list(set(posterior_history[-99:])))<10
            if (ii>2000) and (ii%1000 == 0):
                flat = np.mean(posterior_history[-999:]) < np.mean(posterior_history[-1990:-999])
            else:
                flat = False
            if always_rejecting or flat:
                jump_sd *= .9
        # TODO: write down itermediate chains in case of a crash... also re-read if we restart. Good for debugging purposes.
    return pd.DataFrame(outdicts)


def get_test_loss(n_iters, seed, holdout, shrinkage, params, obs, 
                  forecast_priors, ignore_vent):
    return chain(n_iters = n_iters, seed = seed, params=params, 
                 obs=obs, shrinkage=shrinkage, holdout=holdout,
                 forecast_priors = forecast_priors, ignore_vent = ignore_vent)["test_loss"]


def do_chains(n_iters, 
              params, 
              obs, 
              best_penalty, 
              sample_obs, 
              holdout, 
              n_chains, 
              forecast_priors, 
              parallel,
              ignore_vent):
    tuples_for_starmap = [(i, params, obs, n_iters, best_penalty, holdout, \
                           forecast_priors, sample_obs, ignore_vent) \
                          for i in range(n_chains)]
    # get the final answer based on the best penalty
    if parallel:
        pool = mp.Pool(mp.cpu_count())
        chains = pool.starmap(chain, tuples_for_starmap)
        pool.close()
    else:
        chains = map(lambda x: chain(*x), tuples_for_starmap)
    df = pd.concat(chains, ignore_index=True)
    return df



def main():
    # if __name__ == "__main__":
        # n_chains = 8
        # n_iters = 3000
        # penalty = .25
        # fit_penalty = False
        # sample_obs = False
        # as_of_days_ago = 0
        # census_ts = pd.read_csv(path.join(f"~/projects/chime_sims/data/", f"PAH_ts.csv"), encoding = "latin")
        # # impute vent with the proportion of hosp.  this is a crude hack
        # census_ts.loc[census_ts.vent.isna(), "vent"] = census_ts.hosp.loc[
        #     census_ts.vent.isna()
        # ] * np.mean(census_ts.vent / census_ts.hosp)
        # # import parameters
        # params = pd.read_csv(path.join(f"/Users/crandrew/projects/chime_sims/data/", f"PAH_parameters.csv"), encoding = "latin")
        # flexible_beta = True
        # fit_penalty = True
        # y_max = None
        # figdir = f"/Users/crandrew/projects/chime_sims/output/foo/"
        # outdir = f"/Users/crandrew/projects/chime_sims/output/"
        # burn_in = 2000
        # prefix = ""
        # reopen_day = 100
        # reopen_speed = .1
        # reopen_cap = .5
        
        # forecast_change_prior_mean = 0
        # forecast_change_prior_sd = -99920
        # forecast_priors = dict(mu = forecast_change_prior_mean,
        #                         sig = forecast_change_prior_sd)
        # ignore_vent = True
    # else:
    p = ArgParser()
    p.add("-c", "--my-config", is_config_file=True, help="config file path")
    p.add("-P", "--prefix", help="prefix for old-style inputs")
    p.add("-p", "--parameters", help="the path to the parameters csv")
    p.add("-t", "--ts", help="the path to the time-series csv")
    p.add("-C", "--n_chains", help="number of chains to run", default=8, type=int)
    p.add(
        "-i",
        "--n_iters",
        help="number of iterations to run per chain",
        default=5000,
        type=int,
    )
    p.add(
        "-f",
        "--fit_penalty",
        action="store_true",
        help="fit the penalty based on the last week of data",
    )
    p.add(
        "--penalty",
        help="penalty factor used for shrinkage (0.05 - 1)",
        default=0.05,
        type=float,
    )
    p.add(
        "-s",
        "--sample_obs",
        action="store_true",
        help="adds noise to the values in the time-series",
    )
    p.add("-o", "--out", help="output directory")
    p.add(
        "-a",
        "--as_of",
        default=0,
        help="number of days in the past to project from",
        type=int,
    )
    p.add(
        "-b",
        "--flexible_beta",
        action="store_true",
        help="flexible, vs simple, logistic represetation of beta",
    )
    p.add("-v", "--verbose", action="store_true", help="verbose output")
    p.add(
        "-B",
        "--burn_in",
        type=int,
        help="how much of the burn-in to discard",
        default = 2000
    )
    p.add(
        "-d",
        "--n_days",
        help="make a census/admits plot out to n_days",
        type=int,
        action="append",
    )
    p.add("-y", "--y_max", help="max y-scale for the census graph", type=int)
    p.add(
        "-pp",
        "--plot_pairs",
        action="store_true",
        help="Plot posterior samples in a pair-plot grid",
    )
    p.add(
        "--reopen_day",
        type=int,
        help="day at which to commence evaluating the reopen function",
        default = 8675309
    )
    p.add(
        "--reopen_speed",
        type=float,
        help="how fast to reopen",
        default = 0.1
    )
    p.add(
        "--reopen_cap",
        type=float,
        help="how much reopening to allow",
        default = 1.0
    )
    p.add(
        "--forecast_change_prior_mean",
        type=float,
        help="prior on how much the census will change over the next week, in percent",
        default = 0
    )
    p.add(
        "--forecast_change_prior_sd",
        type=float,
        help="strength of prior on how much the census will change over the next week, in percent",
        default = -9999.9
    )
    p.add(
        "--save_chains",
        action="store_true",
        help="store the chains?  It'll make it take longer, as there is a lot of info in them.",
    )
    p.add(
        "--ignore_vent",
        action="store_true",
        help="don't fit to vent, multiply the likelihood by zero",
    )

    options = p.parse_args()
        
    prefix = options.prefix
    n_chains = options.n_chains
    n_iters = options.n_iters
    penalty = options.penalty
    fit_penalty = options.fit_penalty
    sample_obs = options.sample_obs
    as_of_days_ago = options.as_of
    flexible_beta = options.flexible_beta
    burn_in = options.burn_in
    y_max = options.y_max
    reopen_day = options.reopen_day
    reopen_speed = options.reopen_speed
    reopen_cap = options.reopen_cap
    forecast_priors = dict(mu = options.forecast_change_prior_mean,
                           sig = options.forecast_change_prior_sd)
    save_chains = options.save_chains
    ignore_vent = options.ignore_vent

    if flexible_beta:
        print("doing flexible beta")
    else:
        print('doing logistic')
        
    dir = get_dir_name(options)
    print(dir)

    census_ts, params = get_inputs(options)

    if census_ts is None or params is None:
        print("You must specify either --prefix or --parameters and --ts")
        print(p.format_help())
        exit(1)
        
    if not fit_penalty:
        assert penalty >= 0.05 and penalty < 1

    outdir = path.join(dir, "output")
    makedirs(outdir)
    figdir = path.join(dir, "figures")
    makedirs(figdir)
    paramdir = path.join(dir, "parameters")
    makedirs(paramdir)

    write_inputs(options, paramdir, census_ts, params)
## start here when debug
    nobs = census_ts.shape[0] - as_of_days_ago

    # expand out the spline terms and append them to params
    # also add the number of observations, as i'll need this for evaluating the knots
    # finally, estimate the scaling factors for the design matrix
    if flexible_beta == True:
        beta_spline_power = int(params.loc[params.param == "beta_spline_power", 'base'])
        beta_splines = pd.DataFrame([{
            "param": f"beta_spline_coef_{i}",
            'base':0,
            "distribution":"norm",
            "p1":0,
            "p2":float(params.p2.loc[params.param == 'beta_spline_prior']**beta_spline_power),
            'description':'spile term for beta'
            } for i in range(int(params.base.loc[params.param == "beta_spline_dimension"]))])
        nobsd = pd.DataFrame(dict(param = 'nobs', base = nobs, 
                                  distribution = "constant", p1 = np.nan, 
                                  p2 = np.nan, 
                                  description = 'number of observations(days)'), 
                             index = [0])
        # create the design matrix in order to compute the scaling factors
        # this is critical to make the prior on design matrix flexibility invariant to the scaling of the features
        beta_k = int(params.loc[params.param == "beta_spline_dimension", 'base'])
        knots = np.linspace(0, nobs-nobs/beta_k/2, beta_k)
        X = np.stack([power_spline(day, knots, beta_spline_power, xtrim = nobs) for day in range(nobs)])
        Xmu = np.mean(X, axis = 0)
        Xsig = np.std(X, axis = 0)
        Xscale = pd.DataFrame(dict(param = ['Xmu', 'Xsig'], 
                                   base = [Xmu, Xsig], 
                                   distribution = "constant", 
                                   p1 = [np.nan, np.nan], 
                                   p2 = [np.nan, np.nan], 
                                   description = ['','']))
        params = pd.concat([params, beta_splines, nobsd, Xscale])
        # set the ununsed ones to constant
        params.loc[params.param.isin(['logistic_k', 
                                      'logistic_L', 
                                      'logistic_x0',
                                      'beta_spline_power',
                                      'beta_spline_prior',
                                      'beta_spline_dimension']), 'distribution'] = "constant"
    

    # rolling window variance
    rwstd = []
    for i in range(nobs):
        y = census_ts.hosp[:i][-7:]
        rwstd.append(np.std(y))
    census_ts["hosp_rwstd"] = np.nan
    census_ts.loc[range(nobs), "hosp_rwstd"] = rwstd

    rwstd = []
    for i in range(nobs):
        y = census_ts.vent[:i][-7:]
        rwstd.append(np.std(y))
    census_ts["vent_rwstd"] = np.nan
    census_ts.loc[range(nobs), "vent_rwstd"] = rwstd

    if sample_obs:
        fig = plt.figure()
        plt.plot(census_ts.vent, color="red")
        plt.fill_between(
            x=list(range(nobs)),
            y1=census_ts.vent + 2 * census_ts.vent_rwstd,
            y2=census_ts.vent - 2 * census_ts.vent_rwstd,
            alpha=0.3,
            lw=2,
            edgecolor="k",
        )
        plt.title("week-long rolling variance")
        fig.savefig(path.join(f"{figdir}", f"observation_variance.pdf"))

    if fit_penalty:
        pen_vec = np.linspace(0.05, 0.5, 10)
        tuples_for_starmap = [(n_iters, i, 7, j, params, census_ts, forecast_priors) \
                              for i in range(n_chains) for j in pen_vec]
        pool = mp.Pool(mp.cpu_count())
        shrinkage_chains = pool.starmap(get_test_loss, tuples_for_starmap)
        pool.close()
        # put together the mp results
        chain_dict = {i: [] for i in pen_vec}
        for i in range(len(tuples_for_starmap)):
            chain_dict[tuples_for_starmap[i][3]] += shrinkage_chains[i][
                burn_in:
            ].tolist()  # get the penalty value

        mean_test_loss = [np.mean(np.array(chain_dict[i])) for i in pen_vec]

        fig = plt.figure()
        plt.plot(pen_vec, mean_test_loss)
        plt.fill_between(
            x=pen_vec,
            y1=[float(np.quantile(chain_dict[i][1000:], [0.025])) for i in pen_vec],
            y2=[float(np.quantile(chain_dict[i][1000:], [0.975])) for i in pen_vec],
            alpha=0.3,
            lw=2,
            edgecolor="k",
        )
        plt.xlabel("penalty factor")
        plt.ylabel("log10(test MSE)")
        fig.savefig(path.join(f"{figdir}", f"shrinkage_grid_GOF.pdf"))
        # identify the best penalty
        best_penalty = pen_vec[np.argmin(mean_test_loss)]
    elif penalty < 1:
        best_penalty = penalty
    # fit the actual chains
    df = do_chains(n_iters = n_iters, 
                   params = params, 
                   obs = census_ts, 
                   best_penalty = best_penalty, 
                   sample_obs = sample_obs, 
                   holdout = as_of_days_ago,
                   n_chains = n_chains,
                   forecast_priors = forecast_priors,
                   parallel=True,
                   ignore_vent = ignore_vent)
    if save_chains:
        df.to_json(path.join(f"{outdir}", "chains.json.bz2"), orient="records", lines=True)

    # process the output
    burn_in_df = df.loc[(df.iter <= burn_in)]
    df = df.loc[(df.iter > burn_in)]
    
    # do the SD plot
    SD_plot(census_ts, params, df, figdir, prefix if prefix is not None else "")
    
    ## SEIR plot
    SEIR_plot(df=df, 
              first_day = census_ts[census_ts.columns[0]].values[0], 
              howfar = 200, 
              figdir = figdir, 
              prefix = prefix if prefix is not None else "",
              as_of_days_ago = as_of_days_ago,
              census_ts = census_ts)

    
    
    ## Rt plot
    Rt_plot(df=df, 
              first_day = census_ts[census_ts.columns[0]].values[0], 
              howfar = 200, 
              figdir = figdir, 
              prefix = prefix if prefix is not None else "",
              params = params,
              census_ts = census_ts)

    # make predictive plot
    n_days = [30, 90, 180]
    if options.n_days:
        n_days = options.n_days

    first_day = census_ts[census_ts.columns[0]].values[0]
    for howfar in n_days:
        plt_predictive(
            df,
            first_day,
            census_ts,
            figdir,
            as_of_days_ago,
            howfar=howfar,
            prefix=prefix if prefix is not None else "",
            y_max=y_max,
            hosp_capacity=None,
            vent_capacity=None,
        )

    # reopening
    colors = ['blue', 'green', 'orange', 'red', 'yellow', 'cyan']
    reopen_day_gap = math.ceil((200-reopen_day)/len(colors))
    reopen_days = np.arange(reopen_day, 199, reopen_day_gap)
    qmats = []    
    for day in reopen_days:
        pool = mp.Pool(mp.cpu_count())
        reop = pool.starmap(reopen_wrapper, [(df.iloc[i], day, reopen_speed, reopen_cap) for i in range(df.shape[0])])
        pool.close()
        reop = np.stack(reop)
        reopq = np.quantile(reop, [.05, .25, .5, .75, .95], axis = 0)
        qmats.append(reopq)

    dates = pd.date_range(f"{first_day}", periods=201, freq="d")
    fig = plt.figure()
    for i in range(len(reopen_days)):
        plt.plot_date(dates, qmats[i][2, :], "-", 
                      label=f"re-open after {reopen_days[i]} days",
                      color = colors[i])
        plt.fill_between(x = dates,
                         y1 = qmats[i][1,:], y2 = qmats[i][3,:], 
                         alpha = .2, color = colors[i])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.title(f"Reopening scenario, {int(reopen_speed*100)}% per day up to {int(reopen_cap*100)}% social distancing")
    fig.autofmt_xdate()
    fig.savefig(path.join(f"{figdir}", f"{prefix}reopening_scenarios.pdf"))
     
    

    mk_projection_tables(df, first_day, outdir)

    toplot = df[
        [
            "beta",
            "hosp_prop",
            "ICU_prop",
            "vent_prop",
            "hosp_LOS",
            "ICU_LOS",
            "vent_LOS",
            "incubation_days",
            "recovery_days",
            "logistic_k",
            "logistic_x0",
            "logistic_L",
            "nu",
        ]
    ]

    pspace = np.linspace(0.001, 0.999, 1000)

    fig, ax = plt.subplots(figsize=(8, 40), ncols=1, nrows=len(toplot.columns))
    for i in range(len(toplot.columns)):
        cname = toplot.columns[i]
        if params.loc[params.param == cname, "distribution"].iloc[0] == "gamma":
            x = sps.gamma.ppf(
                pspace,
                params.loc[params.param == cname, "p1"],
                0,
                params.loc[params.param == cname, "p2"],
            )
            y = sps.gamma.pdf(
                x,
                params.loc[params.param == cname, "p1"],
                0,
                params.loc[params.param == cname, "p2"],
            )
        elif params.loc[params.param == cname, "distribution"].iloc[0] == "beta":
            x = sps.beta.ppf(
                pspace,
                params.loc[params.param == cname, "p1"],
                params.loc[params.param == cname, "p2"],
            )
            y = sps.beta.pdf(
                x,
                params.loc[params.param == cname, "p1"],
                params.loc[params.param == cname, "p2"],
            )
        ax[i].plot(x, y, label="prior")
        ax[i].hist(toplot[cname], density=True, label="posterior", bins=30)
        ax[i].set_xlabel(params.loc[params.param == cname, "description"].iloc[0])
        ax[i].legend()
    plt.tight_layout()
    fig.savefig(path.join(f"{figdir}", 
                          f"{prefix if prefix is not None else ''}marginal_posteriors_v2.pdf"))

    if options.plot_pairs:
        #  Make a pair plot for diagnosing posterior dependence
        plt_pairplot_posteriors(toplot, figdir, prefix=prefix)



    if options.verbose:
        print(f"Output directory: {dir}")
    else:
        print(dir)



if __name__ == "__main__":
    main()
