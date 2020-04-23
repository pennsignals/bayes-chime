"""Utilities for the normal CHIME Bayes module
"""
from typing import TypeVar, Union

from numpy import exp

FloatLike = TypeVar("FloatLike")  # Floats or integers
FloatLikeArray = TypeVar("FloatLikeArray")  # Arrays of floats or integers

NormalDistVar = TypeVar("NormalDistVar")  # Normally distributed random var
NormalDistArray = TypeVar("NormalDistArray")  # Array of Normally dist random var

FloatOrDistVar = Union[FloatLike, NormalDistVar]
FloatOrDistArray = Union[FloatLikeArray, NormalDistArray]


def logistic_fcn(  # pylint: disable=C0103
    x: FloatOrDistArray, L: FloatOrDistVar, k: FloatOrDistVar, x0: FloatOrDistVar,
) -> FloatOrDistArray:
    """Computes `L / (1 + exp(-k(x-x0)))`.
    """
    return L / (1 + exp(-k * (x - x0)))


def one_minus_logistic_fcn(  # pylint: disable=C0103
    x: FloatOrDistArray, L: FloatOrDistVar, k: FloatOrDistVar, x0: FloatOrDistVar,
) -> FloatOrDistArray:
    """Computes `1 - L / (1 + exp(-k(x-x0)))`.
    """
    return 1 - logistic_fcn(x, L, k, x0)
