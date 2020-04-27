"""Tests for SIR model in this repo
* Compares conserved quantities
* Compares model against Penn CHIME w/wo social policies
* Checks logistic policies in extreme limit
"""
from typing import Tuple
from datetime import date, timedelta

from pytest import fixture

from pandas import DataFrame, Series, DatetimeIndex
from pandas.testing import assert_frame_equal, assert_series_equal

from penn_chime.model.parameters import Parameters, Disposition
from penn_chime.model.sir import (
    Sir,
    sim_sir,
    calculate_dispositions,
    calculate_admits,
    calculate_census,
)

from bayes_chime.normal.models import SIRModel
from bayes_chime.normal.utilities import one_minus_logistic_fcn

PENN_CHIME_COMMIT = "188c35be9561164bedded4a8071a320cbde0d2bc"

COLS_TO_COMPARE = [
    "susceptible",
    "infected",
    "recovered",
    "hospital_admits",
    # Does not compare census as this repo uses the exponential distribution
]
COLUMN_MAP = {
    "hospitalized": "hospital_admits",
}


@fixture(name="penn_chime_setup")
def fixture_penn_chime_setup() -> Tuple[Parameters, Sir]:
    """Initializes penn_chime parameters and SIR model
    """
    p = Parameters(
        current_hospitalized=69,
        date_first_hospitalized=date(2020, 3, 7),
        doubling_time=None,
        hospitalized=Disposition.create(days=7, rate=0.025),
        icu=Disposition.create(days=9, rate=0.0075),
        infectious_days=14,
        market_share=0.15,
        n_days=100,
        population=3600000,
        recovered=0,
        relative_contact_rate=0.3,
        ventilated=Disposition.create(days=10, rate=0.005),
    )
    return p, Sir(p)


@fixture(name="penn_chime_raw_df_no_policy")
def fixture_penn_chime_raw_df_no_policy(penn_chime_setup) -> DataFrame:
    """Runs penn_chime SIR model for no social policies
    """
    p, simsir = penn_chime_setup

    n_days = simsir.raw_df.day.max() - simsir.raw_df.day.min()
    policies = [(simsir.beta, n_days)]
    raw = sim_sir(
        simsir.susceptible,
        simsir.infected,
        p.recovered,
        simsir.gamma,
        -simsir.i_day,
        policies,
    )
    calculate_dispositions(raw, simsir.rates, market_share=p.market_share)
    calculate_admits(raw, simsir.rates)
    calculate_census(raw, simsir.days)

    raw_df = DataFrame(raw)
    raw_df.index = simsir.raw_df.date

    return raw_df.fillna(0)


@fixture(name="sir_data_wo_policy")
def fixture_sir_data_wo_policy(penn_chime_setup, penn_chime_raw_df_no_policy):
    """Provides data for local sir module
    """
    p, simsir = penn_chime_setup
    raw_df = penn_chime_raw_df_no_policy
    day0 = raw_df.iloc[0].fillna(0)

    total = day0.susceptible + day0.infected + day0.recovered

    pars = {
        "beta": simsir.beta * total,  # This repo uses S/total in sir
        "gamma": simsir.gamma,
        "initial_susceptible": day0.susceptible,
        "initial_infected": day0.infected,
        "initial_hospital": day0.hospitalized,
        "initial_recovered": day0.recovered,
        "hospital_probability": simsir.rates["hospitalized"],
    }
    x = {
        "dates": DatetimeIndex(raw_df.index),
        "hospital_length_of_stay": p.dispositions["hospitalized"].days,
        "market_share": p.market_share,
    }
    return x, pars


@fixture(name="sir_data_w_policy")
def fixture_sir_data_w_policy(penn_chime_setup):
    """Provides data for local sir module with implemented policies
    """
    p, simsir = penn_chime_setup
    raw_df = simsir.raw_df.set_index("date")
    day0 = raw_df.iloc[0].fillna(0)

    total = day0.susceptible + day0.infected + day0.recovered

    pars = {
        "beta": simsir.beta * total,  # This repo uses S/total in sir
        "gamma": simsir.gamma,
        "initial_susceptible": day0.susceptible,
        "initial_infected": day0.infected,
        "initial_hospital": day0.hospitalized,
        "initial_recovered": day0.recovered,
        "hospital_probability": simsir.rates["hospitalized"],
    }
    x = {
        "dates": DatetimeIndex(raw_df.index),
        "hospital_length_of_stay": p.dispositions["hospitalized"].days,
        "market_share": p.market_share,
    }
    return x, pars


def test_conserved_n(sir_data_wo_policy):
    """Checks if S + I + R is conserved for local SIR
    """
    x, pars = sir_data_wo_policy
    sir_model = SIRModel()

    n_total = 0
    for key in sir_model.compartments:
        n_total += pars[f"initial_{key}"]

    predictions = sir_model.propagate_uncertainties(x, pars)

    n_computed = predictions[sir_model.compartments].sum(axis=1)
    n_expected = Series(data=[n_total] * len(n_computed), index=n_computed.index)

    assert_series_equal(n_expected, n_computed)


def test_sir_vs_penn_chime_no_policies(penn_chime_raw_df_no_policy, sir_data_wo_policy):
    """Compares local SIR against penn_chime SIR for no social policies
    """
    x, pars = sir_data_wo_policy

    sir_model = SIRModel()
    predictions = sir_model.propagate_uncertainties(x, pars)

    assert_frame_equal(
        penn_chime_raw_df_no_policy.rename(columns=COLUMN_MAP)[COLS_TO_COMPARE],
        predictions[COLS_TO_COMPARE],
    )


def test_sir_vs_penn_chime_w_policies(penn_chime_setup, sir_data_w_policy):
    """Compares local SIR against penn_chime SIR for with social policies
    """
    p, sir = penn_chime_setup
    x, pars = sir_data_w_policy

    policies = sir.gen_policy(p)
    new_policy_date = x["dates"][0] + timedelta(days=policies[0][1])
    beta0, beta1 = policies[0][0], policies[1][0]

    def update_parameters(ddate, **pars):  # pylint: disable=W0613
        pars["beta"] = (beta0 if ddate < new_policy_date else beta1) * p.population
        return pars

    sir_model = SIRModel(update_parameters=update_parameters)
    predictions = sir_model.propagate_uncertainties(x, pars)

    assert_frame_equal(
        sir.raw_df.set_index("date")
        .fillna(0)
        .rename(columns=COLUMN_MAP)[COLS_TO_COMPARE],
        predictions[COLS_TO_COMPARE],
    )


def test_sir_logistic_policy(penn_chime_setup, sir_data_w_policy):
    """Compares local SIR against penn_chime SIR for implemented social policies
    where policies are implemented as a logistic function
    """
    p, sir = penn_chime_setup
    x, pars = sir_data_w_policy

    policies = sir.gen_policy(p)

    # Set up logistic function to match policies (Sharp decay)
    pars["beta"] = policies[0][0] * p.population
    ## This are new parameters needed by one_minus_logistic_fcn
    pars["L"] = 1 - policies[1][0] / policies[0][0]
    pars["x0"] = policies[0][1] - 0.5
    pars["k"] = 1.0e7

    def update_parameters(ddate, **kwargs):
        xx = (ddate - x["dates"][0]).days
        ppars = kwargs.copy()
        ppars["beta"] = kwargs["beta"] * one_minus_logistic_fcn(
            xx, L=kwargs["L"], k=kwargs["k"], x0=kwargs["x0"],
        )
        return ppars

    sir_model = SIRModel(update_parameters=update_parameters)
    predictions = sir_model.propagate_uncertainties(x, pars)

    assert_frame_equal(
        sir.raw_df.set_index("date")
        .rename(columns=COLUMN_MAP)[COLS_TO_COMPARE]
        .fillna(0),
        predictions[COLS_TO_COMPARE],
    )


def test_sir_type_conversion(sir_data_w_policy):
    """Compares local SIR run with set gamma vs set with recovery_days
    """
    x, pars = sir_data_w_policy

    sir_model = SIRModel()
    predictions = sir_model.propagate_uncertainties(x, pars)

    pars["recovery_days"] = 1 / pars.pop("gamma")
    new_predictions = sir_model.propagate_uncertainties(x, pars)

    assert_frame_equal(
        predictions, new_predictions,
    )
