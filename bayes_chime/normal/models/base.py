"""Helper functions to utilize SIR like models
"""
from typing import Dict, Generator, List, Callable, Optional

from abc import ABC, abstractmethod

from datetime import date as Date
from datetime import timedelta

from pandas import DataFrame, DatetimeIndex, infer_freq

from bayes_chime.normal.utilities import (
    FloatLike,
    NormalDistVar,
    FloatOrDistVar,
    NormalDistArray,
)


class CompartmentModel(ABC):
    """Abstract implementation of SIR like compartment model

    Attributes:
        model_parameters: A list of all parameters needed to run a simmulation
            (used by `simulation_step` or `post_process_simulation`).
            These parameters must be present after `parse_input` is run.
        optional_parameters: Further parameters which extend model predictions but
            are not required. E.g., to do conversions between parameters.
        compartments: These are the compartments needed by the model.
            E.g., susceptible, infected and recovered for standard SIR.
    """

    # ----------------------------------------
    # Below you can find methods to overload
    # ----------------------------------------

    model_parameters: List[str] = ["dates"]
    optional_parameters: List[str] = []
    compartments: List[str] = []

    def parse_input(  # pylint: disable=R0201
        self, **pars: Dict[str, FloatOrDistVar]
    ) -> Dict[str, FloatOrDistVar]:
        """Parses parameters before fitting. This should include, e.g., type conversions

        By default, checks dates and adds frequency.
        """
        dates = pars["dates"]
        if not isinstance(dates, DatetimeIndex):
            raise TypeError("Dates must be of type DatetimeIndex")

        if not dates.freq:
            dates.freq = infer_freq(dates)
            pars["dates"] = dates

        pars["days_per_step"] = pars["dates"].freq / timedelta(days=1)

        return pars

    def post_process_simulation(  # pylint: disable=R0201, W0613, C0103
        self, df: DataFrame, **pars: Dict[str, FloatOrDistVar]
    ) -> DataFrame:
        """Processes the final simulation result. This can add, e.g., new columns
        """
        return df

    @abstractmethod
    def simulation_step(
        self, data: Dict[str, NormalDistVar], **pars: Dict[str, FloatOrDistVar]
    ):
        """This function implements the actual simulation

        Arguments:
            data: The compartments for each iteration
            pars: Model parameters

        Returns:
            Updated compartments and optionally additional information like change
            from last iteration.
        """
        return data

    # ----------------------------------------------------------
    # This part should be fixed unless you add functionality
    # ----------------------------------------------------------

    def __init__(
        self,
        fit_columns: Optional[List[str]] = None,
        update_parameters: Callable[
            [Date, Dict[str, FloatOrDistVar]], Dict[str, FloatOrDistVar]
        ] = None,
    ):
        """Initializes the compartment model
        update function

        Arguments:
            fit_columns: When calling fit_fcn, this will only return specified column.
                This should be used when only a subset of the simulation parameters
                should be fit.
            update_parameters: This function allows to update effective model parameters
                based on the number of model iterations. It takes the number of iteration
                and all model (initial) parameters as input. It should return updated
                parameters and defaults to no parameter updates. This can be used to
                implement social distancing.

        Note:
            If the update_parameters method requires additional arguments, they must be
            passed to the respective model function calls.
        """
        self.fit_columns = fit_columns
        self.update_parameters = (
            update_parameters
            if update_parameters is not None
            else lambda date, **pars: pars
        )

    def propagate_uncertainties(
        self, meta_pars: Dict[str, FloatLike], dist_pars: Dict[str, NormalDistVar]
    ) -> DataFrame:
        """Propagates uncertainties through simmulation

        Arguments:
            meta_pars: Fixed model meta parameters
            dist_pars: Variable model prior parameters

        Returns:
            DataFrame containing simulation data
        """
        pars = self.parse_input(**meta_pars, **dist_pars)

        df = DataFrame(data=self._iterate_simulation(**pars)).set_index("date")

        return self.post_process_simulation(df, **pars)

    def _iterate_simulation(
        self, **pars: Dict[str, FloatOrDistVar],
    ) -> Generator[Dict[str, NormalDistVar], None, None]:
        """Iterates model to build up SIR data

        Initial data is at day zero (no step).

        Arguments:
            n_iter: Number of iterations
            pars: Model meta and flexible parameters
        """
        data = {
            compartment: pars["initial_{compartment}".format(compartment=compartment)]
            for compartment in self.compartments
        }
        for date in pars["dates"]:
            data["date"] = date
            yield data
            inp_pars = self.update_parameters(date, **pars)
            data = self.simulation_step(data, **inp_pars)

    def fit_fcn(  # pylint: disable=C0103
        self, xx: Dict[str, FloatLike], pp: Dict[str, NormalDistVar]
    ) -> NormalDistArray:
        """Wrapper for propagate_uncertainties used for lsqfit.nonlinear_fit function

        Arguments:
            xx: Fixed model meta parameters
            pp: Variable model prior parameters

        Returns:
            Array of `fit_columns` columns without first row (inital data).
        """
        df = self.propagate_uncertainties(xx, pp).drop(0)
        if self.fit_columns:
            df = df[self.fit_columns]

        return df.values if df.values.shape[0] > 1 else df.values.flatten()

    def check_call(  # pylint: disable=C0103
        self,
        xx: Dict[str, FloatLike],
        yy: NormalDistArray,
        pp: Dict[str, NormalDistVar],
    ):
        """Checks if model meta parameters and priors are set as expected to use fitter.

        Checks:
            * Non-overlapping model parameters
            * Data retrun shape
            * Data retun types

        Raises:
            Specific error messages if not set up correctly
        """
        common_pars = xx.keys().intersection(pp.keys())
        if common_pars:
            raise KeyError(
                "Fixed and variable model paramers have shared variables:"
                " {common_pars}. This is not allowed.".format(common_pars=common_pars)
            )

        yy_fit = self.fit_fcn(xx, pp)
        if not yy_fit.shape == yy.shape:
            raise ValueError(
                "Fit function return has different shape as data"
                " fit: {fit_shape} data: {data_shape}.".format(
                    fit_shape=yy_fit.shape, data_shape=yy.shape
                )
            )

        for n_el, (y_fit, y_data) in enumerate(zip(yy_fit, yy)):
            if not isinstance(yy_fit, type(yy)):
                raise TypeError(
                    "Element {n_el} of fit has different type as data:".format(
                        n_el=n_el
                    )
                    + "\t{y_fit} -> {y_fit_type}!= {y_data} -> {y_data_type}".format(
                        y_fit_type=type(y_fit),
                        y_fit=y_fit,
                        y_data_type=type(y_data),
                        y_data=y_data,
                    )
                )
