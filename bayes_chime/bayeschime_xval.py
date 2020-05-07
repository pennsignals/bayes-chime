


'''
Bayes chime cross-validation:
Run regular chime-sims then bayes chime, holding out data along the way to check performance

'''
import pandas as pd
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000
import os
from typing import Dict, Tuple

from argparse import ArgumentParser

from datetime import date as Date
from datetime import timedelta

from pandas import DataFrame, date_range, read_csv
from scipy.stats import expon

from gvar import dump, mean, sdev

from gvar._gvarcore import GVar  # pylint: disable=E0611
from gvar import gvar
from lsqfit import nonlinear_fit, empbayes_fit
import copy

from bayes_chime.normal.utilities import (
    FloatOrDistVar,
    FloatLike,
    NormalDistVar,
    NormalDistArray,
    one_minus_logistic_fcn,
)

from bayes_chime.normal.models import SEIRModel
from bayes_chime.normal.scripts.utils import (
    DEBUG,
    read_parameters,
    read_data,
    dump_results,
    get_logger,
)
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import probplot

from _01_GOF_sims import do_chains

datadir = f"{os.getcwd()}/data/"
outdir = f"{os.getcwd()}/output/"
figdir = f"{os.getcwd()}/figures/"



def logisitic_social_policy(
    date: Date, **kwargs: Dict[str, FloatOrDistVar]
) -> Dict[str, FloatOrDistVar]:
    """Updates beta parameter as a function of time by multiplying base parameter
    with 1 - logistic function.

    Relevant keys are:
        * dates
        * beta
        * logistic_L
        * logistic_k
        * logistic_x0
    """
    xx = (date - kwargs["dates"][0]).days
    ppars = kwargs.copy()
    ppars["beta"] = kwargs["beta"] * one_minus_logistic_fcn(
        xx, L=kwargs["logistic_L"], k=kwargs["logistic_k"], x0=kwargs["logistic_x0"],
    )
    return ppars


def prepare_model_parameters(
    parameters: Dict[str, FloatOrDistVar], data: DataFrame
) -> Tuple[Dict[str, FloatLike], Dict[str, NormalDistVar]]:
    """Prepares model input parameters and returns independent and dependent parameters

    Also shifts back simulation to start with only exposed people.
    """

    # Set up fit parameters
    ## Dependent parameters which will be fitted
    pp = {key: val for key, val in parameters.items() if isinstance(val, GVar)}
    ## Independent model meta parameters
    xx = {key: val for key, val in parameters.items() if key not in pp}

    # This part ensures that the simulation starts with only exposed persons
    ## E.g., we shift the simulation backwards such that exposed people start to
    ## become infected
    xx["offset"] = int(
        expon.ppf(0.99, 1 / pp["incubation_days"].mean)
    )  # Enough time for 95% of exposed to become infected
    pp["logistic_x0"] += xx["offset"]

    ## Store the actual first day
    xx["day0"] = data.index.min()
    ## And start earlier in time
    xx["dates"] = date_range(
        xx["day0"] - timedelta(xx["offset"]), freq="D", periods=xx["offset"]
    ).union(data.index)

    ## Thus, all compartment but exposed and susceptible are 0
    for key in ["infected", "recovered", "icu", "vent", "hospital"]:
        xx[f"initial_{key}"] = 0

    pp["initial_exposed"] = (
        xx["n_hosp"] / xx["market_share"] / pp["hospital_probability"]
    )
    xx["initial_susceptible"] -= pp["initial_exposed"].mean

    return xx, pp


def get_yy(data: DataFrame, **err: Dict[str, FloatLike]) -> NormalDistArray:
    """Converts data to gvars by adding uncertainty:

    yy_sdev = yy_mean * rel_err + min_er
    """
    return gvar(
        [data["hosp"].values, data["vent"].values],
        [
            data["hosp"].values * err["hosp_rel"] + err["hosp_min"],
            data["vent"].values * err["vent_rel"] + err["vent_min"],
        ],
    ).T


def bayes_xval(days_withheld = 7, which_hospital = "HUP"):
    try:
        parameters = read_parameters(f"{datadir}{which_hospital}_parameters.csv")
        data = read_data(f"{datadir}{which_hospital}_ts.csv")[:-days_withheld]
        test_set = pd.read_csv(f"{datadir}{which_hospital}_ts.csv")[-days_withheld:]
        test_set.date = test_set.date.astype("datetime64[ns]")
        model = SEIRModel(
            fit_columns=["hospital_census", "vent_census"],
            update_parameters=logisitic_social_policy,
        )
        
        xx, pp = prepare_model_parameters(parameters, data)
        model.fit_start_date = xx["day0"]
    
        fit_kwargs = lambda error_infos: dict(
            data=(xx, get_yy(data, hosp_rel=0, vent_rel=0, **error_infos)),
            prior=pp,
            fcn=model.fit_fcn,
            debug=True,
        )
        fit, xx["error_infos"] = empbayes_fit(
            {"hosp_min": 10, "vent_min": 1}, fit_kwargs
        )
        # extend by 60 days
        xx["dates"] = xx["dates"].union(
            date_range(xx["dates"].max(), freq="D", periods=60)
        )
        prediction_df = model.propagate_uncertainties(xx, fit.p)
        prediction_df.index = prediction_df.index.round("H")
        
        # drop the index
        prediction_df = prediction_df.reset_index()
        prediction_df['hmu'] = prediction_df.hospital_census.apply(lambda x: float(str(x).split("(")[0]))
        prediction_df['hsig'] = prediction_df.hospital_census.apply(lambda x: float(str(x).split("(")[1][:-1]) if "(" in str(x) else float(x))
        prediction_df['vmu'] = prediction_df.vent_census.apply(lambda x: float(str(x).split("(")[0]))
        prediction_df['vsig'] = prediction_df.vent_census.apply(lambda x: float(str(x).split("(")[1][:-1]) if "(" in str(x) else float(x))
    
        # merge
        mm = prediction_df.merge(test_set, how = 'left')
        tomerge = copy.deepcopy(data)
        tomerge.columns = ["obs_"+ i for i in tomerge.columns]
        tomerge.reset_index(inplace = True)
        mm = mm.merge(tomerge, how = 'outer')
        
        # compute simple msfe
        hMSFE = np.mean((mm.hmu - mm.hosp)**2)
        vMSFE = np.mean((mm.vmu - mm.vent)**2)
        loss_approx = (hMSFE+vMSFE)/2
    
        # now run MCMC
        params_raw = pd.read_csv(f"{datadir}{which_hospital}_parameters.csv")
        df = do_chains(n_iters = 9000, params = params_raw, 
                       obs = read_data(f"{datadir}{which_hospital}_ts.csv"), 
                       best_penalty = None,  sample_obs = False, 
                       holdout = days_withheld, n_chains = 1, parallel = False)
        df = df.loc[df.iter>1000]
        arrs = np.stack([df.arr.iloc[i] for i in range(df.shape[0])])
        arrs_test = arrs[:,data.shape[0]:(data.shape[0]+days_withheld),:]
        median_pred = np.median(arrs_test, axis = 0)
        loss_mcmc = (np.mean((median_pred[:,3] - test_set.hosp)**2) \
                     + np.mean((median_pred[:,5] - test_set.vent)**2))/2        # output
            
        # prediction quantiles:  the proportion of times the prediction is greater than the mean
        
        hq_m = [(arrs_test[:,day,3] > test_set.hosp.iloc[day]).mean() for day in range(days_withheld)]
        vq_m = [(arrs_test[:,day,5] > test_set.vent.iloc[day]).mean() for day in range(days_withheld)]
        
        mmtail = mm.tail(days_withheld)
        hq_n = [(np.random.normal(mmtail.hmu.iloc[day], mmtail.hsig.iloc[day], 10000) > test_set.hosp.iloc[day]).mean() for day in range(days_withheld)]
        vq_n = [(np.random.normal(mmtail.vmu.iloc[day], mmtail.vsig.iloc[day], 10000) > test_set.vent.iloc[day]).mean() for day in range(days_withheld)]
        
        plotr = dict(hq_m = hq_m,
                     vq_m = vq_m,
                     hq_n = hq_n,
                     vq_n = vq_n)
        
        
               
        resh_n = mm.hmu[:len(data)+len(test_set)] - np.array(data.hosp.tolist() + test_set.hosp.tolist())
        resh_m = np.median(arrs[:, :len(data)+len(test_set), 3], axis = 0) - np.array(data.hosp.tolist() + test_set.hosp.tolist())
        resv_n = mm.vmu[:len(data)+len(test_set)] - np.array(data.vent.tolist() + test_set.vent.tolist())
        resv_m = np.median(arrs[:, :len(data)+len(test_set), 5], axis = 0) - np.array(data.vent.tolist() + test_set.vent.tolist())

        
        plotq = dict(resh_n = mm.hmu[:len(data)+len(test_set)] - np.array(data.hosp.tolist() + test_set.hosp.tolist()),
                     resh_m = np.median(arrs[:, :len(data)+len(test_set), 3], axis = 0) - np.array(data.hosp.tolist() + test_set.hosp.tolist()),
                     resv_n = mm.vmu[:len(data)+len(test_set)] - np.array(data.vent.tolist() + test_set.vent.tolist()),
                     resv_m = np.median(arrs[:, :len(data)+len(test_set), 5], axis = 0) - np.array(data.vent.tolist() + test_set.vent.tolist())
                     )
                    
        
        out = dict(which_hospital = which_hospital,
                   days_out = days_withheld,
                   loss_mcmc = loss_mcmc,
                   loss_approx = loss_approx,
                   plotq = plotq,
                   plotr = plotr)
        return out
    except Exception as e:
        print(e)






def plotr(d):
    fig, ax = plt.subplots(nrows = 1, ncols = 2)
    ax[0].plot(d['hq_m'], label = "mcmc")
    ax[0].plot(d['hq_n'], label = "gaussian")
    ax[0].legend()
    ax[0].set_ylabel("quantile")
    ax[0].set_xlabel("day of past week")
    
    ax[1].plot(d['vq_m'], label = "mcmc")
    ax[1].plot(d['vq_n'], label = "gaussian")
    ax[1].legend()
    ax[1].set_ylabel("quantile")
    ax[1].set_xlabel("day of past week")

    fig.suptitle(f"{d['which_hospital']}")
    plt.tight_layout()



def plotq(d):
    fig, ax = plt.subplots(nrows = 2, ncols = 2)
    probplot(d['resh_m'], dist = "norm", plot = ax[0,0])
    ax[0,0].set_title(f"Residuals, hospital, mcmc")
    probplot(d['resh_n'], dist = "norm", plot = ax[0,1])
    ax[0,1].set_title(f"Residuals, hospital, gaussian")
    probplot(d['resv_m'], dist = "norm", plot = ax[1,0])
    ax[1,0].set_title(f"Residuals, vent, mcmc")
    probplot(d['resv_n'], dist = "norm", plot = ax[1,1])
    ax[1,1].set_title(f"Residuals, vent, gaussian")
    fig.suptitle(f"{d['which_hospital']}")
    plt.tight_layout()


def main():

    tuples_for_starmap = [(7, j) for j in ['PMC', "LGH", "HUP", "CCH", 'PAH', 'MCP']]
    
    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())
    outdicts = pool.starmap(bayes_xval, tuples_for_starmap)
    pool.close()
        
    
    for i in outdicts:
        plotr(i['plotr'])
        plotq(i['plotq'])
    
    
    
    csvout = pd.DataFrame(outdicts)
    csvout.to_csv(f"{outdir}xval_results.csv")
        
    
    csvout = pd.read_csv(f"{outdir}xval_results.csv")
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows = len(csvout.which_hospital.unique()))
    fig.set_size_inches(8.5, 25.5)
    for i in range(len(csvout.which_hospital.unique())):
        ho = csvout.which_hospital.unique()[i]
        ax[i].plot(csvout.loc[csvout.which_hospital == ho, 'days_out'], 
                 csvout.loc[csvout.which_hospital == ho, 'loss_mcmc'], 
                 label = 'MCMC ')
        ax[i].plot(csvout.loc[csvout.which_hospital == ho, 'days_out'], 
                 csvout.loc[csvout.which_hospital == ho, 'loss_approx'], 
                 label = 'gaussian approximation')
        ax[i].legend()
        ax[i].set_title(ho)
        ax[i].set_ylabel("Mean squared forecast error")
        ax[i].set_xlabel("Number of days to predict")
        fig.tight_layout()
        # fig.savefig(f"{figdir}xval_plots.pdf")
    
if __name__ == "__main__":
    main()