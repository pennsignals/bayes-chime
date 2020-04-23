"""Helper functions to utilize SIR like models
"""
from typing import Dict, Generator, List, Callable

from abc import ABC, abstractmethod

from numpy import arange
from pandas import DataFrame

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
        compartments: These are the compartments needed by the model.
            E.g., susceptible, infected and recovered for standard SIR.
    """

    model_parameters: List[str] = []
    compartments: List[str] = []

    def __init__(
        self,
        fit_columns: List[str],
        update_parameters: Callable[
            [int, Dict[str, FloatOrDistVar]], Dict[str, FloatOrDistVar]
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
            The update_parameters method should not update parameters in place.
        """
        self.fit_columns = fit_columns
        self.update_parameters = (
            update_parameters
            if Dict[str, FloatOrDistVar] is not None
            else lambda nn, pars: pars
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

        df = DataFrame(data=self._iterate_simulation(**pars), index=pars["dates"])

        return self.post_process_simulation(df, **pars)

    def parse_input(  # pylint: disable=R0201
        self, **pars: Dict[str, FloatOrDistVar]
    ) -> Dict[str, FloatOrDistVar]:
        """Parses parameters before fitting. This should include, e.g., type conversions
        """
        return pars

    def post_process_simulation(  # pylint: disable=R0201, W0613, C0103
        self, df: DataFrame, **pars: Dict[str, FloatOrDistVar]
    ) -> DataFrame:
        """Processes the final simulation result. This can add, e.g., new columns
        """
        return df

    def _iterate_simulation(
        self, n_iter: int, **pars: Dict[str, FloatOrDistVar],
    ) -> Generator[Dict[str, NormalDistVar]]:
        """Iterates model to build up SIR data

        Initial data is at day zero (no step).

        Arguments:
            n_iter: Number of iterations
            pars: Model meta and flexible parameters
        """
        for nn in arange(n_iter):
            yield data
            inp_pars = self.update_parameters(nn, **pars)
            data = self.simulation_step(data, **inp_pars)

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
        return df.values

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