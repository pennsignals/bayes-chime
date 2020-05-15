
"""Command line interface script for running a Bayesian fit from command line
bayeschime -m -p data/Downtown_parameters.csv -d data/Downtown_ts.csv -y data/data_errors.csv -b flexible_beta
"""
from typing import Dict, Tuple

from argparse import ArgumentParser

from datetime import date as Date
from datetime import timedelta

from pandas import DataFrame, date_range, read_csv
from scipy.stats import expon

from gvar._gvarcore import GVar  # pylint: disable=E0611
from gvar import gvar
from lsqfit import nonlinear_fit, empbayes_fit


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
    mse,
)
import copy
import numpy as np
import multiprocessing as mp
import pandas as pd


LOGGER = get_logger(__name__)


def parse_args():
    """Arguments of the command line script
    """
    parser = ArgumentParser()

    parser.add_argument(
        "-p",
        "--parameter-file",
        help="File to read prior parameters from. This file is required.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--data-file",
        help="File to read data from. Must have columns `hosp` and `vent`."
        " This file is required.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-y",
        "--data-error-file",
        help="File to read data error policy from."
        " This specifies relative and absolute errors of `hosp` and `vent`."
        " E.g., y_sdev = y_mean * rel_err + abs_arr."
        " If not given, employs empirical Bayes to abs_arr only to estimate errors.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="The directory to dump results into."
        " This will dump a pickle file (read in by gvar.load('fit.pickle')) which"
        " completely determines the fit,"
        " a prediction csv and a pdf plot. Default: %(default)s",
        type=str,
        default="output",
    )
    parser.add_argument(
        "-e",
        "--extend-days",
        help="Extend prediction by number of days. Default: %(default)s",
        type=int,
        default=30,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Add more verbosity",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-b",
        "--beta",
        help="Which function to use for beta over time; logistic, or flexible beta",
        type=str,
        default="logistic_social_policy",
    )
    parser.add_argument(
        "-k",
        "--spline_dimension",
        help = "The number of knots to add to each spline term",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-P",
        "--spline_power",
        help="Exponent on the truncated power spline",
        type = int,
        default = 2,)
    parser.add_argument(
        "-X",
        "--cross_validate",
        help="flag to ignore variances on spline terms, and rather cross-validate to get them instead",
        action="store_true",
        default=True,
    )
    args = parser.parse_args()
    return args


def logistic_social_policy(
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


def flexible_beta(
        date: Date, **kwargs: Dict[str, FloatOrDistVar]
) -> Dict[str, FloatOrDistVar]:
    '''
    Implements flexible social distancing
    '''
    xx = (date - kwargs["dates"][0]).days
    ppars = kwargs.copy()
    X = power_spline(xx, kwargs['knots'], kwargs['spline_power'])
    ppars["beta"] = kwargs["beta"] * (1-1/(1+np.exp(kwargs['beta_intercept'] + X@kwargs['beta_splines'])))
    return ppars


def power_spline(x, knots, n):
    if x > max(knots): #trim the ends of the spline to prevent nonsense extrapolation
        x = max(knots)+1
    spl = x - np.array(knots)
    spl[spl<0] = 0
    # # to flatten trends past the last day, set trends equal to max of knots, plus one
    # spl[spl>(max(knots)+1)] = max(knots)+1
    return spl**n



def prepare_model_parameters(
    parameters: Dict[str, FloatOrDistVar], data: DataFrame, 
    beta_fun, splines, spline_power
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
    # pp["logistic_x0"] += xx["offset"]
    xx['beta_fun'] = beta_fun
    xx['knots'] = splines
    xx['spline_power'] = spline_power

    ## Store the actual first day and the actual last day
    xx["day0"] = data.index.min()
    xx["day-1"] = data.index.max()

    ## And start earlier in time
    xx["dates"] = date_range(
        xx["day0"] - timedelta(xx["offset"]), freq="D", periods=xx["offset"]
    ).union(data.index)

    # initialize the spline parameters on the flexible beta
    if xx['beta_fun'] == "flexible_beta":
        pp['beta_splines'] = gvar([pp['pen_beta'].mean for i in range(len(xx['knots']))],
                             [pp['pen_beta'].sdev for i in range(len(xx['knots']))])
        pp.pop("pen_beta")
        pp.pop('logistic_k')
        pp.pop('logistic_x0')
        pp.pop('logistic_L')
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




def xval_wrapper(pen, win, parameter_file_path, splines, spline_power, 
                 data_file_path, data_error_file_path, k):
    try:
        parameters = read_parameters(parameter_file_path)
        data = read_data(data_file_path)
        tr = data[:win]
        val = data[win:(win+7)]

        mi = SEIRModel(
            fit_columns=["hospital_census", "vent_census"],
            update_parameters=flexible_beta
        )
        xx, pp = prepare_model_parameters(parameters = parameters, data = tr, 
                                          beta_fun = 'flexible_beta', splines = splines,
                                          spline_power = spline_power)      
        pp['beta_splines'] = gvar([0 for i in range(k)], [pen for i in range(k)])
        mi.fit_start_date = xx["day0"]
        xx["error_infos"] = (
            read_csv(data_error_file_path).set_index("param")["value"].to_dict()
        )
        fit = nonlinear_fit(
            data=(xx, get_yy(tr, **xx["error_infos"])),
            prior=pp,
            fcn=mi.fit_fcn,
            debug=False,
        )        
        # detect and handle degenerate fits
        # THIS IS A TEMPORARY HACK
        splinecoefvec = np.array([fit.p['beta_splines'][i].mean for i in range(len(fit.p['beta_splines']))])
        cv = np.std(splinecoefvec)/np.mean(splinecoefvec)
        if cv < .1:
            MSE = -9999
            error = "degenerate fit"
        else:
            xx = fit.x.copy()
            xx["dates"] = xx["dates"].union(
                date_range(xx["dates"].max(), freq="D", periods=8)
            )
            prediction_df = mi.propagate_uncertainties(xx, fit.p) 
            prediction_df.index = prediction_df.index.round("H")
            mg = val.merge(prediction_df, left_index = True, right_index = True)
            # scaling
            hosp = (mg.hosp-np.mean(mg.hosp))/np.std(mg.hosp)
            hosp_hat = (mg.hospital_census.apply(lambda x: x.mean)-np.mean(mg.hosp))/np.std(mg.hosp)
            vent = (mg.vent-np.mean(mg.vent))/np.std(mg.vent)
            vent_hat = (mg.vent_census.apply(lambda x: x.mean)-np.mean(mg.vent))/np.std(mg.vent)
            MSE = mse(hosp, hosp_hat) + mse(vent, vent_hat) 
            error = ""
        return dict(mse = MSE,
                    pen = pen,
                    win = win, 
                    error = error)
    except Exception as e:
        return dict(mse = -9999,
                    pen = pen,
                    win = win, 
                    error = e)


def main():
    """Executes the command line script
    """
    
    if __name__ == "__main__":
        parameter_file_path = 'data/foo.csv'
        parameters = read_parameters('data/foo.csv')
        data_file_path = 'data/HUP_ts.csv'
        data = read_data(data_file_path)
        error_file_path = 'data/data_errors.csv'
        model = SEIRModel(
            fit_columns=["hospital_census", "vent_census"],
            update_parameters=flexible_beta
        )
        xval = True
        k = 10
        spline_power = 2
        splines = np.linspace(0, 
                        data.shape[0]-5,
                        k).astype(int)
        win = 40
        pen = .002
        beta_fun = 'flexible_beta'
        pd.options.display.max_rows = 4000
        pd.options.display.max_columns = 4000
    else:
        args = parse_args()
        #
        data_file_path = args.data_file
        parameter_file_path = args.parameter_file
        beta_fun = args.beta
        spline_power = args.spline_power
        xval = args.cross_validate if args.beta == "flexible_beta" else False
        error_file_path = args.data_error_file
        k = args.spline_dimension

        if args.verbose:
            for handler in LOGGER.handlers:
                handler.setLevel(DEBUG)
    
        LOGGER.debug("Received arguments:\n%s", args)
    
        parameters = read_parameters(parameter_file_path)
        LOGGER.debug("Read parameters:\n%s", parameters)
    
        data = read_data(data_file_path)
        LOGGER.debug("Read data:\n%s", data)

        model = SEIRModel(
            fit_columns=["hospital_census", "vent_census"],
            update_parameters=flexible_beta if beta_fun == "flexible_beta" \
                                            else logistic_social_policy,
        )


        # parse the splines
        # TODO:  note this will need to be generalized once we've got more features time-varying
        if k > 0:
            splines = np.arange(0, 
                                data.shape[0],
                                int(data.shape[0]/k))
        else:
            splines = -99
            assert args.beta != "flexible_beta", "You need to specify some splines with '-k <spline dimension> if you're using flexible beta"

    ## CROSS VALIDATION
    if xval is True:
        print("Doing rolling-window cross-validation")
        assert error_file_path is not None, "Haven't yet implemented cross-validation for empirical bayes.  Please supply a data error file (i.e.: `-y data/data_errors.csv`)"
        # loop through windows, and in each one, forecast one week out.  
        penvec = 10**np.linspace(-10, 5, 16)

        winstart = list(range(data.shape[0]-14, (data.shape[0]-7)))
        tuples_for_starmap = [(p,
                               w,
                               parameter_file_path,
                               splines, 
                               k, 
                               data_file_path, 
                               error_file_path, 
                               k) for p in penvec for w in winstart]
        
        pool = mp.Pool(mp.cpu_count())
        xval_results = pool.starmap(xval_wrapper, tuples_for_starmap)
        pool.close()
        xval_df = pd.DataFrame(xval_results)
        # remove errors
        errors = (xval_df.mse == -9999).sum()
        # assert errors < xval_df.shape[0]*.2, "Lot's of errors when doing cross-validation.  Breaking here rather than returning unreliable results."
        xval_df = xval_df.loc[xval_df.mse >0]
        xval_df['rmse'] = xval_df.mse**.5
        penframe = xval_df.groupby(['pen']).agg({'rmse':['mean', 'std']}, as_index = False).reset_index()
        penframe.columns = ['pen', 'mu', 'sig']

        
        best_penalty = penframe.pen.loc[penframe.mu == min(penframe.mu)].iloc[0]
        print(f"The best prior sd on the splines is {best_penalty}.  Don't forget to look at the plot of cross-validation statistics (in the output directory) to make sure that there's nothing wacky going on.")
        parameters['pen_beta'] = gvar(0,best_penalty)


    degen_flag = True
    while degen_flag:
        xx, pp = prepare_model_parameters(parameters = parameters, data = data, 
                                          beta_fun = beta_fun, splines = splines,
                                          spline_power = spline_power)
        LOGGER.debug("Parsed model meta pars:\n%s", xx)
        LOGGER.debug("Parsed model priors:\n%s", pp)
        model.fit_start_date = xx["day0"]
    
        # If empirical bayes is selected to fit the data, this also returns the fit object
        LOGGER.debug("Starting fit")
        if args.data_error_file:
            xx["error_infos"] = (
                read_csv(error_file_path).set_index("param")["value"].to_dict()
            )
            
            LOGGER.debug("Using y_errs from file:\n%s", xx["error_infos"])
            fit = nonlinear_fit(
                data=(xx, get_yy(data, **xx["error_infos"])),
                prior=pp,
                fcn=model.fit_fcn,
                # debug=args.verbose,
            )
        else:
            LOGGER.debug("Employing empirical Bayes to infer y-errors")
            # This fit varies the size of the y-errors of hosp_min and vent_min
            # to optimize the description of the data (logGBF)
            fit_kwargs = lambda error_infos: dict(
                data=(xx, get_yy(data, hosp_rel=0, vent_rel=0, **error_infos)),
                prior=pp,
                fcn=model.fit_fcn,
                debug=args.verbose,
            )
            fit, xx["error_infos"] = empbayes_fit(
                {"hosp_min": 10, "vent_min": 1}, fit_kwargs
            )
            LOGGER.debug("Empbayes y_errs are:\n%s", xx["error_infos"])
        # check for degeneracy
        splinecoefvec = np.array([fit.p['beta_splines'][i].mean for i in range(len(fit.p['beta_splines']))])
        cv = np.std(splinecoefvec[1:])/np.mean(splinecoefvec[1:])
        BI_update = (fit.p['beta_intercept'] - parameters['beta_intercept']).mean
        coef_OM_range = np.ptp(np.log10(np.abs(splinecoefvec)))
        if (cv < .1) | (BI_update**2<.1)| (coef_OM_range > 2):
            print('the best prior sd on the splines led to a degenerate fit.  trimming it by one order of magnitude')
            curr = np.log10(best_penalty)
            assert curr > -5, "degenerate solutions all the way down.  Something is broken."
            best_penalty = 10**(curr-1)
            print(f"new best prior sd on the splines is {best_penalty}")
            parameters['pen_beta'] = gvar(0,best_penalty)
        else:
            degen_flag = False

    LOGGER.info("Fit result:\n%s", fit)

    dump_results(args.output_dir, fit=fit, model=model, extend_days=args.extend_days)
    LOGGER.debug("Dumped results to:\n%s", args.output_dir)


if __name__ == "__main__":
    main()


#         import matplotlib.pyplot as plt
#         # plt.scatter(xval_df.win, xval_df.rmse)
#         # plt.scatter(np.log10(xval_df.pen), xval_df.rmse)
#         plt.plot(np.log10(penframe.pen), penframe.mu)
#         plt.fill_between(x = np.log10(penframe.pen),
#                          y1 = penframe.mu+penframe.sig,
#                          y2 = penframe.mu-penframe.sig,
#                          alpha = .3)
        


# def plot_beta(fit):
#     x = np.arange(0, len(fit.x['dates']),1)
#     beta = []
#     for i in x:
#         X = power_spline(i, fit.x['knots'], fit.x['spline_power'])
#         b = fit.p['beta'] * (1-1/(1+np.exp(fit.p['beta_intercept'] + X@fit.p['beta_splines'])))
#         beta.append(b)
#     muvec = np.array([i.mean for i in beta])
#     sdvec = np.array([i.sdev for i in beta])
#     plt.plot(x, muvec)
#     plt.fill_between(x = x,
#                      y1 = muvec+sdvec*1.96,
#                      y2 = muvec - sdvec*1.96,
#                      alpha = .3)
#     plt.xlabel(f"Days since {min(fit.x['dates'])}")
#     plt.ylabel('beta')

# plot_beta(fit)

#     @@ next:  roll with penalty that works best.  
#     @@ dump plots of cross-validation statistics        
#     @@ make sure cross-validation isn't too wacky
