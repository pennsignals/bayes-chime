"""Utility function for command line script
"""
from typing import Dict

from logging import getLogger, StreamHandler, Formatter, DEBUG, INFO

from pandas import read_csv, DataFrame

from bayes_chime.normal.utilities import FloatOrDistVar
from bayes_chime.normal.fitting import fit_norm_to_prior_df

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


def dump_results(*args, **kwargs):
    """
    """
