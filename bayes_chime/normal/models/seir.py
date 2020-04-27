"""Implementation of SIR model
"""
from typing import Dict, List

from bayes_chime.normal.utilities import FloatOrDistVar, NormalDistVar
from bayes_chime.normal.models.sir import SIRModel


class SEIRModel(SIRModel):
    """Basic SEIR model
    """

    model_parameters: List[str] = SIRModel.model_parameters + [
        "alpha",  # or incubation_days
    ]
    optional_parameters: List[str] = SIRModel.optional_parameters + [
        "incubation_days",
    ]
    compartments: List[str] = SIRModel.compartments + [
        "exposed",
    ]

    def parse_input(  # pylint: disable=R0201
        self, **pars: Dict[str, FloatOrDistVar]
    ) -> Dict[str, FloatOrDistVar]:
        """Parses dates, gamma and beta parameters if applicable.
        """
        pars = super().parse_input(**pars)

        if "alpha" not in pars:
            pars["alpha"] = 1 / (pars["incubation_days"] / pars["days_per_step"])

        return pars

    @staticmethod
    def simulation_step(
        data: Dict[str, NormalDistVar], **pars: Dict[str, FloatOrDistVar]
    ):
        """Executes SIR step and patches results such that each component is larger zero.

        Arguments:
            data:
                susceptible: Susceptible population
                exposed: Exposed poplation
                infected: Infected population
                recovered: Recovered population
            pars:
                beta: Growth rate for infected
                alpha: Incubation rate for infected
                gamma: Recovery rate for infected
                nu: changes effect of susceptible for exposed to `(S/N) ** nu`

        Returns:
            Updated compartments and optionally additional information like change
            from last iteration.
        """
        susceptible = data["susceptible"]
        exposed = data["exposed"]
        infected = data["infected"]
        recovered = data["recovered"]

        total = susceptible + exposed + infected + recovered

        d_se = pars["beta"] * (susceptible / total) ** pars["nu"] * infected
        d_ei = pars["alpha"] * exposed
        d_ir = pars["gamma"] * infected

        susceptible -= d_se
        exposed += d_se - d_ei
        infected += d_ei - d_ir
        recovered += d_ir

        susceptible = max(susceptible, 0)
        exposed = max(exposed, 0)
        infected = max(infected, 0)
        recovered = max(recovered, 0)

        rescale = total / (susceptible + exposed + infected + recovered)

        out = {
            "susceptible": susceptible * rescale,
            "exposed": exposed * rescale,
            "infected": infected * rescale,
            "recovered": recovered * rescale,
            "exposed_new": d_se * rescale,
            "infected_new": d_ei * rescale,
            "recovered_new": d_ir * rescale,
        }

        return out
