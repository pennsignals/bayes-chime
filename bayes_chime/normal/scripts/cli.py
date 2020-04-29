"""Command line interface script for running a Bayesian fit from command line
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
)


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
    args = parser.parse_args()

    return args


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


def main():
    """Executes the command line script
    """
    args = parse_args()

    if args.verbose:
        for handler in LOGGER.handlers:
            handler.setLevel(DEBUG)

    LOGGER.debug("Received arguments:\n%s", args)

    parameters = read_parameters(args.parameter_file)
    LOGGER.debug("Read parameters:\n%s", parameters)

    data = read_data(args.data_file)
    LOGGER.debug("Read data:\n%s", data)

    model = SEIRModel(
        fit_columns=["hospital_census", "vent_census"],
        update_parameters=logisitic_social_policy,
    )

    xx, pp = prepare_model_parameters(parameters, data)
    LOGGER.debug("Parsed model meta pars:\n%s", xx)
    LOGGER.debug("Parsed model priors:\n%s", pp)
    model.fit_start_date = xx["day0"]

    # If empirical bayes is selected to fit the data, this also returns the fit object
    LOGGER.debug("Starting fit")
    if args.data_error_file:
        xx["error_infos"] = (
            read_csv(args.data_error_file).set_index("param")["value"].to_dict()
        )
        LOGGER.debug("Using y_errs from file:\n%s", xx["error_infos"])
        fit = nonlinear_fit(
            data=(xx, get_yy(data, **xx["error_infos"])),
            prior=pp,
            fcn=model.fit_fcn,
            debug=args.verbose,
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

    LOGGER.info("Fit result:\n%s", fit)

    dump_results(args.output_dir, fit=fit, model=model, extend_days=args.extend_days)
    LOGGER.debug("Dumped results to:\n%s", args.output_dir)


if __name__ == "__main__":
    main()



'''
Plan:  get his thing working in a single script
Compare it against regular bayeschime
'''


