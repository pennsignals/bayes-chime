"""Tests for SEIR model in this repo
* Compares conserved quantities
* Compares model against SEIR wo social policies in limit to SIR
"""
from pytest import fixture

from pandas import Series
from pandas.testing import assert_frame_equal, assert_series_equal

from bayes_chime.normal.models import SIRModel, SEIRModel

from tests.normal.models.sir_test import (  # pylint: disable=W0611
    fixture_sir_data_wo_policy,
    fixture_penn_chime_setup,
    fixture_penn_chime_raw_df_no_policy,
)

COLS_TO_COMPARE = [
    "susceptible",
    "infected",
    "recovered",
    # Does not compare census as this repo uses the exponential distribution
]

PENN_CHIME_COMMIT = "188c35be9561164bedded4a8071a320cbde0d2bc"


@fixture(name="seir_data")
def fixture_seir_data(sir_data_wo_policy):
    """Returns data for the SIHR model
    """
    x, p = sir_data_wo_policy
    pp = p.copy()
    xx = x.copy()
    pp["alpha"] = 0.5
    pp["nu"] = 1
    pp["initial_exposed"] = 0

    return xx, pp


def test_conserved_n(seir_data):
    """Checks if S + E + I + R is conserved for SEIR
    """
    x, pars = seir_data

    n_total = 0
    for key in SEIRModel.compartments:
        n_total += pars[f"initial_{key}"]

    seir_model = SEIRModel()
    predictions = seir_model.propagate_uncertainties(x, pars)

    n_computed = predictions[SEIRModel.compartments].sum(axis=1)
    n_expected = Series(data=[n_total] * len(n_computed), index=n_computed.index)

    assert_series_equal(n_expected, n_computed)


def test_compare_sir_vs_seir(sir_data_wo_policy, seir_data, monkeypatch):
    """Checks if SEIR and SIR return same results if the code enforces

    * alpha = gamma
    * E = 0
    * dI = dE
    """
    x_sir, pars_sir = sir_data_wo_policy
    x_seir, pars_seir = seir_data

    pars_seir["alpha"] = pars_sir["gamma"]  # will be done by hand

    def mocked_seir_step(data, **pars):
        data["exposed"] = 0
        new_data = SEIRModel.simulation_step(data, **pars)
        new_data["infected"] += new_data["exposed_new"]
        return new_data

    seir_model = SEIRModel()
    monkeypatch.setattr(seir_model, "simulation_step", mocked_seir_step)

    sir_model = SIRModel()
    predictions_sir = sir_model.propagate_uncertainties(x_sir, pars_sir)
    predictions_seir = seir_model.propagate_uncertainties(x_seir, pars_seir)

    assert_frame_equal(
        predictions_sir[COLS_TO_COMPARE], predictions_seir[COLS_TO_COMPARE],
    )
