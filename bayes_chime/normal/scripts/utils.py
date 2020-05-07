"""Utility function for command line script
"""
from typing import Dict, TypeVar

from os import path, makedirs

from datetime import datetime
from logging import getLogger, StreamHandler, Formatter, DEBUG, INFO

from pandas import read_csv, DataFrame, date_range, Series

from gvar import dump, mean, sdev

from bayes_chime.normal.utilities import FloatOrDistVar
from bayes_chime.normal.models.base import CompartmentModel
from bayes_chime.normal.fitting import fit_norm_to_prior_df
from bayes_chime.normal.plotting import plot_fit
import numpy as np

PARAMETER_MAP = {
    "hosp_prop": "hospital_probability",
    "hosp_LOS": "hospital_length_of_stay",
    "ICU_prop": "icu_probability",
    "ICU_LOS": "icu_length_of_stay",
    "vent_prop": "vent_probability",
    "vent_LOS": "vent_length_of_stay",
    "region_pop": "initial_susceptible",
    "mkt_share": "market_share",
}

Fit = TypeVar("NonLinearFit")


def get_logger(name: str):
    """Sets up logger
    """

    logger = getLogger(name)
    logger.setLevel(DEBUG)

    if not logger.handlers:
        ch = StreamHandler()
        ch.setLevel(INFO)
        formatter = Formatter("[%(asctime)s - %(name)s - %(levelname)s] %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def read_parameters(file_name: str) -> Dict[str, FloatOrDistVar]:
    """Reads in parameter file, fits a normal distribution and returns parameters needed
    for normal module.

    Arguments:
        file_name: Path to the parameter file
    """
    df = read_csv(file_name)
    parameters = fit_norm_to_prior_df(df)
    return {PARAMETER_MAP.get(key, key): val for key, val in parameters.items()}


def read_data(file_name: str) -> DataFrame:
    """Reads in data file, fits a normal distribution and returns keys needed for
    normal module.

    Expects columns "hosp", "vent".

    Arguments:
        file_name: Path to the data file
    """
    df = (
        read_csv(file_name, parse_dates=["date"])
        .dropna(how="all", axis=1)
        .fillna(0)
        .set_index("date")
        .astype(int)
    )
    for col in ["hosp", "vent"]:
        if not col in df.columns:
            raise KeyError(f"Could not locate column {col}.")

    return df


def dump_results(
    output_dir: str, fit: Fit, model: CompartmentModel, extend_days: int = 30
):
    """Exports fit and model to pickle file, saves forecast as csv and saves plot
    """
    now_dir = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    dir_name = path.join(output_dir, now_dir)
    if not path.exists(dir_name):
        makedirs(dir_name, exist_ok=True)

    # Write fit to file. Can be read with gvar.load(file)
    dump(
        {"model": model, "fit": fit}, outputfile=path.join(dir_name, "fit.pickle"),
    )

    # Extend day range for next steps
    xx = fit.x.copy()
    if extend_days:
        xx["dates"] = xx["dates"].union(
            date_range(xx["dates"].max(), freq="D", periods=extend_days)
        )

    # Generate new prediction
    prediction_df = model.propagate_uncertainties(xx, fit.p)
    prediction_df.index = prediction_df.index.round("H")
    if model.fit_start_date:
        prediction_df = prediction_df.loc[model.fit_start_date :]

    # Dump forecast
    (
        prediction_df.stack()
        .apply(lambda el: Series(dict(mean=mean(el), sdev=sdev(el))))
        .reset_index(level=1)
        .rename(columns={"level_1": "kind"})
        .to_csv(path.join(dir_name, "forecast.csv"))
    )

    # Dump plot
    fig = plot_fit(
        prediction_df,
        columns=(
            ("hospital_census", "vent_census"),
            ("hospital_admits", "vent_admits"),
        ),
        data={key: fit.y.T[ii] for ii, key in enumerate(model.fit_columns)},
    )
    fig.savefig(path.join(dir_name, "forecast.pdf"), bbox_inches="tight")

def mse(x,y):
    return np.mean((x-y)**2)