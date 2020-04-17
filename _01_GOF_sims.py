from copy import deepcopy
from datetime import datetime
from os import getcwd, path, mkdir
from string import ascii_letters, digits
import json
import multiprocessing as mp

from configargparse import ArgParser
from git import Repo
from scipy import stats as sps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _99_shared_functions import SIR_from_params, qdraw, jumper
from utils import beta_from_q

LET_NUMS = pd.Series(list(ascii_letters) + list(digits))
PARAMDIR = None
CENSUS_TS = None
PARAMS = None
NOBS = None
N_ITERS = None

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
    mkdir(outdir)
    return outdir


def get_inputs(options):
    census_ts, params = None, None
    if options.prefix is not None:
        prefix = options.prefix
        datadir = path.join(f"{getcwd()}", "data")
        # import the census time series and set the zero day to be the first instance of zero
        census_ts = pd.read_csv(path.join(f"{datadir}", f"{prefix}_ts.csv"))
        # impute vent with the proportion of hosp.  this is a crude hack
        census_ts.loc[census_ts.vent.isna(), "vent"] = census_ts.hosp.loc[
                                                           census_ts.vent.isna()
                                                       ] * np.mean(census_ts.vent / census_ts.hosp)
        # import parameters
        params = pd.read_csv(path.join(f"{datadir}", f"{prefix}_parameters.csv"))
    if options.parameters is not None:
        params = pd.read_csv(options.parameters)
    if options.ts is not None:
        census_ts = pd.read_csv(options.ts)
        # impute vent with the proportion of hosp.  this is a crude hack
        census_ts.loc[census_ts.vent.isna(), "vent"] = census_ts.hosp.loc[
                                                           census_ts.vent.isna()
                                                       ] * np.mean(census_ts.vent / census_ts.hosp)
    return census_ts, params


def write_inputs(options):
    with open(path.join(PARAMDIR, "args.json"), "w") as f:
        json.dump(options.__dict__, f)
    CENSUS_TS.to_csv(path.join(PARAMDIR, "census_ts.csv"), index=False)
    PARAMS.to_csv(path.join(PARAMDIR, "params.csv"), index=False)
    with open(path.join(PARAMDIR, "git.sha"), "w") as f:
        f.write(Repo(search_parent_directories=True).head.object.hexsha)

def loglik(r):
    return -len(r) / 2 * (np.log(2 * np.pi * np.var(r))) - 1 / (
            2 * np.pi * np.var(r)
    ) * np.sum(r ** 2)


def do_shrinkage(pos, shrinkage):
    densities = sps.beta.pdf(pos, a=shrinkage[0], b=shrinkage[1])
    regularization_penalty = -np.sum(np.log(densities))
    return regularization_penalty


def eval_pos(pos, shrinkage=None, holdout=0, sample_obs=True):
    """function takes quantiles of the priors and outputs a posterior and relevant stats"""
    draw = SIR_from_params(qdraw(pos, PARAMS))
    obs = deepcopy(CENSUS_TS)
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
                draw["arr"][: (NOBS - holdout), 5] - train.vent.values[:NOBS]
        )  # 5 corresponds with vent census
        if any(residuals_vent == 0):
            residuals_vent[residuals_vent == 0] = 0.01
        sigma2 = np.var(residuals_vent)
        LL += loglik(residuals_vent)

    # loss for hosp
    residuals_hosp = (
            draw["arr"][: (NOBS - holdout), 3] - train.hosp.values[:NOBS]
    )  # 5 corresponds with vent census
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
        posterior -= do_shrinkage(pos, shrinkage)

    out = dict(
        pos=pos,
        draw=draw,
        posterior=posterior,
        residuals_vent=residuals_vent,
        residuals_hosp=residuals_hosp,
    )
    if holdout > 0:
        res_te_vent = draw["arr"][(NOBS - holdout) : NOBS, 5] - test.vent.values[:NOBS]
        res_te_hosp = draw["arr"][(NOBS - holdout) : NOBS, 3] - test.hosp.values[:NOBS]
        test_loss = (np.mean(res_te_hosp ** 2) + np.mean(res_te_vent ** 2)) / 2
        out.update({"test_loss": test_loss})
    return out


def chain(seed, shrinkage=None, holdout=0, sample_obs=False):
    np.random.seed(seed)
    if shrinkage is not None:
        assert (shrinkage < 1) and (shrinkage >= 0.05)
        sq1 = shrinkage / 2
        sq2 = 1 - shrinkage / 2
        shrinkage = beta_from_q(sq1, sq2)
    current_pos = eval_pos(
        np.random.uniform(size=PARAMS.shape[0]),
        shrinkage=shrinkage,
        holdout=holdout,
        sample_obs=sample_obs,
    )
    outdicts = []
    U = np.random.uniform(0, 1, N_ITERS)
    for ii in range(N_ITERS):
        try:
            proposed_pos = eval_pos(
                jumper(current_pos["pos"], 0.1),
                shrinkage=shrinkage,
                holdout=holdout,
                sample_obs=sample_obs,
            )
            p_accept = np.exp(proposed_pos["posterior"] - current_pos["posterior"])
            if U[ii] < p_accept:
                current_pos = proposed_pos

        except Exception as e:
            print(e)
        # append the relevant results
        out = {
            current_pos["draw"]["parms"].param[i]: current_pos["draw"]["parms"].val[i]
            for i in range(PARAMS.shape[0])
        }
        #out.update({"arr": current_pos["draw"]["arr"]})
        out.update({"arr": current_pos["draw"]["arr_stoch"]})
        out.update({"iter": ii})
        out.update({"chain": seed})
        out.update({"posterior": proposed_pos["posterior"]})
        out.update({"offset": current_pos["draw"]["offset"]})
        if holdout > 0:
            out.update({"test_loss": current_pos["test_loss"]})
        outdicts.append(out)
        if shrinkage is None:
            # TODO: write down itermediate chains in case of a crash... also re-read if we restart. Good for debugging purposes.
            if (ii % 1000) == 0:
                print("chain", seed, "iter", ii)
    return pd.DataFrame(outdicts)


def loop_over_shrinkage(seed, holdout=7, shrvec=np.linspace(0.05, 0.95, 10)):
    test_loss = []
    for shr in shrvec:
        chain_out = chain(seed, shr, holdout)
        test_loss.append(chain_out["test_loss"])
    return test_loss


def get_test_loss(seed, holdout, shrinkage):
    return chain(seed, shrinkage, holdout)["test_loss"]


def main():
    global PARAMDIR
    global CENSUS_TS
    global PARAMS
    global NOBS
    global N_ITERS
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

    options = p.parse_args()

    n_chains = options.n_chains
    N_ITERS = options.n_iters
    penalty = options.penalty
    fit_penalty = options.fit_penalty
    sample_obs = options.sample_obs
    as_of_days_ago = options.as_of
    dir = get_dir_name(options)

    if not fit_penalty:
        assert penalty >= 0.05 and penalty < 1

    CENSUS_TS, PARAMS = get_inputs(options)
    if CENSUS_TS is None or PARAMS is None:
        print("You must specify either --prefix or --parameters and --ts")
        print(p.format_help())
        exit(1)

    outdir = path.join(dir, "output")
    mkdir(outdir)
    figdir = path.join(dir, "figures")
    mkdir(figdir)
    PARAMDIR = path.join(dir, "parameters")
    mkdir(PARAMDIR)

    write_inputs(options)

    NOBS = CENSUS_TS.shape[0] - as_of_days_ago

    # rolling window variance
    rwstd = []
    for i in range(NOBS):
        y = CENSUS_TS.hosp[:i][-7:]
        rwstd.append(np.std(y))
    CENSUS_TS["hosp_rwstd"] = np.nan
    CENSUS_TS.loc[range(NOBS), "hosp_rwstd"] = rwstd


    rwstd = []
    for i in range(NOBS):
        y = CENSUS_TS.vent[:i][-7:]
        rwstd.append(np.std(y))
    CENSUS_TS["vent_rwstd"] = np.nan
    CENSUS_TS.loc[range(NOBS), "vent_rwstd"] = rwstd


    if sample_obs:
        fig = plt.figure()
        plt.plot(CENSUS_TS.vent, color="red")
        plt.fill_between(
            x=list(range(NOBS)),
            y1=CENSUS_TS.vent + 2 * CENSUS_TS.vent_rwstd,
            y2=CENSUS_TS.vent - 2 * CENSUS_TS.vent_rwstd,
            alpha=0.3,
            lw=2,
            edgecolor="k",
        )
        plt.title("week-long rolling variance")
        fig.savefig(path.join(f"{figdir}", f"observation_variance.pdf"))



    if fit_penalty:
        pen_vec = np.linspace(0.05, 0.95, 10)
        tuples_for_starmap = [(i, 7, j) for i in range(n_chains) for j in pen_vec]
        pool = mp.Pool(mp.cpu_count())
        shrinkage_chains = pool.starmap(get_test_loss, tuples_for_starmap)
        pool.close()
        # put together the mp results
        chain_dict = {i: [] for i in pen_vec}
        for i in range(len(tuples_for_starmap)):
            chain_dict[tuples_for_starmap[i][2]] += shrinkage_chains[i][
                1000:
            ].tolist()  # get the penalty value

        mean_test_loss = [np.mean(chain_dict[i]) for i in pen_vec]

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
        plt.ylabel("test MSE")
        fig.savefig(path.join(f"{figdir}", f"shrinkage_grid_GOF.pdf"))

        # identify the best penalty
        best_penalty = pen_vec[np.argmin(mean_test_loss)]
    elif penalty < 1:
        best_penalty = penalty

    tuples_for_starmap = [(i, best_penalty, 0, sample_obs) for i in range(n_chains)]

    # get the final answer based on the best penalty
    pool = mp.Pool(mp.cpu_count())
    chains = pool.starmap(chain, tuples_for_starmap)
    pool.close()

    df = pd.concat(chains, ignore_index=True)
    df.to_json(path.join(f"{outdir}", "chains.json.bz2"), orient='records', lines=True)
    print(f"Output directory: {dir}")

if __name__ == '__main__':
    main()

