"""Command line interface script for running a Bayesian fit from command line
"""
from typing import Dict, Tuple, Optional

from argparse import ArgumentParser

from datetime import date as Date
from datetime import timedelta

from pandas import DataFrame, date_range, read_csv
from scipy.stats import expon

from gvar._gvarcore import GVar  # pylint: disable=E0611
from gvar import gvar
from lsqfit import nonlinear_fit


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
    Fit,
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
        help="File to read parameters from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--data-file",
        help="File to read data from. Must have columns `hosp` and `vent`.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-y",
        "--data-error-file",
        help="File to read data error policy from."
        " This specifies relative and absolute errors of `hosp` and `vent`."
        " If not given, employes emperical Bayes to esitmate errors.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="The directory to dump results into.",
        type=str,
        default="output",
    )
    parser.add_argument(
        "-e",
        "--extend-days",
        help="Extend prediction by number of days. Default: %s(default)",
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


def prepare_data(
    data: DataFrame, data_error_file: Optional[str] = None
) -> Tuple[NormalDistArray, Optional[Fit]]:
    """Associates errors to data.

    Errors of data are uncorrelated normal errors with
    std(y) = mean(y) * rel_err + min_err

    If data_error_file is given, parameters are read from file. If not emperical bayes
    is used to infer parameters.
    """
    if not "hosp" in data or not "vent" in data:
        raise KeyError("Data does not contain required columns (hosp and vent).")

    if data_error_file:
        error_infos = read_csv(data_error_file).set_index("param")["value"].to_dict()
        yy = gvar(
            [data["hosp"].values, data["vent"].values],
            [
                data["hosp"].values * error_infos["hosp_rel"] + error_infos["hosp_min"],
                data["vent"].values * error_infos["vent_rel"] + error_infos["vent_min"],
            ],
        ).T
        fit = None
    else:
        ...

    return yy, fit


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

    # If emperical bayes is selected to fit the data, this also returns the fit object
    yy, fit = prepare_data(data, args.data_error_file)
    LOGGER.debug("Prepared fit data:\n%s", yy)
    if not fit:
        fit = nonlinear_fit(data=(xx, yy), prior=pp, fcn=model.fit_fcn, debug=False)

    LOGGER.info("Fit result:\n%s", fit)

    dump_results(args.output_dir, fit=fit, model=model, extend_days=args.extend_days)
    LOGGER.debug("Dumped results to:\n%s", args.output_dir)


if __name__ == "__main__":
    main()
