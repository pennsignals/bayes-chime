"""Implementation of SIR model
"""
from typing import Dict, List

from numpy import log, NaN
from pandas import DataFrame


from bayes_chime.normal.utilities import FloatOrDistVar, NormalDistVar
from bayes_chime.normal.models.base import CompartmentModel


class SIRModel(CompartmentModel):
    """Basic SIR model
    """

    model_parameters: List[str] = [
        "dates",  # DatetimeIndex
        "initial_susceptible",
        "initial_infected",
        "inital_recovered",
        "beta",  # or inital_doubling_time
        "gamma",  # or recovery_days
    ]
    optional_parameters: List[str] = [
        "recovery_days",
        "inital_doubling_time",
        # all keywords below are used to compute hospital admissions and census
        "market_share",
        "initial_hospital",
        "hospital_probability",
        "hospital_length_of_stay",
        "initial_icu",
        "icu_probability",
        "icu_length_of_stay",
        "initial_vent",
        "vent_probability",
        "vent_length_of_stay",
    ]
    compartments: List[str] = ["susceptible", "infected", "recovered"]

    def parse_input(  # pylint: disable=R0201
        self, **pars: Dict[str, FloatOrDistVar]
    ) -> Dict[str, FloatOrDistVar]:
        """Parses dates, gamma and beta parameters if applicable.
        """
        pars = super().parse_input(**pars)

        if "gamma" not in pars:
            pars["gamma"] = 1 / (pars["recovery_days"] / pars["days_per_step"])

        if "beta" not in pars:
            total_population = 0
            for comp in self.compartments:
                total_population += pars["intial_" + comp]

            beta = log(2) / (pars["inital_doubling_time"] / pars["days_per_step"])
            beta += pars["gamma"]
            beta *= total_population / pars["initial_susceptible"]
            pars["beta"] = beta

        return pars

    def post_process_simulation(  # pylint: disable=R0201, W0613, C0103
        self, df: DataFrame, **pars: Dict[str, FloatOrDistVar]
    ) -> DataFrame:
        """Compute admits and census based on exponential LOS distribution if parameters
        present.
        """
        # fill initial hosp admits if present
        df = df.fillna(0)

        # Local infections
        df["infected_new_local"] = df["infected_new"] * pars.get("market_share", NaN)

        dependency = {
            "hospital": "infected_new_local",
            "icu": "hospital_admits",
            "vent": "icu_admits",
        }

        for kind in ["hospital", "icu", "vent"]:
            df[f"{kind}_admits"] = df[dependency[kind]].values * pars.get(
                f"{kind}_probability", NaN
            )

            census = [pars.get(f"initial_{kind}", NaN)]
            for base in df[f"{kind}_admits"].values[1:]:
                census.append(
                    base
                    + (1 - 1 / pars.get(f"{kind}_length_of_stay", NaN)) * census[-1]
                )
            df[f"{kind}_census"] = census

        return df

    def simulation_step(
        self, data: Dict[str, NormalDistVar], **pars: Dict[str, FloatOrDistVar]
    ):
        """Executes SIR step and patches results such that each component is larger zero.

        Arguments:
            data:
                susceptible: Susceptible population
                infected: Infected population
                recovered: Recovered population
            pars:
                beta: Growth rate for infected
                gamma: Recovery rate for infected

        Returns:
            Updated compartments and optionally additional information like change
            from last iteration.
        """
        susceptible = data["susceptible"]
        infected = data["infected"]
        recovered = data["recovered"]

        total = susceptible + infected + recovered

        d_si = pars["beta"] * susceptible / total * infected
        d_ir = pars["gamma"] * infected

        susceptible -= d_si
        infected += d_si - d_ir
        recovered += d_ir

        susceptible = max(susceptible, 0)
        infected = max(infected, 0)
        recovered = max(recovered, 0)

        rescale = total / (susceptible + infected + recovered)

        out = {
            "susceptible": susceptible * rescale,
            "infected": infected * rescale,
            "recovered": recovered * rescale,
            "infected_new": d_si * rescale,
            "recovered_new": d_ir * rescale,
        }

        return out
