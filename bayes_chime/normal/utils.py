"""Utilities for the normal CHIME Bayes module
"""
from typing import TypeVar, Union

from numpy import exp

FloatLike = TypeVar("FloatLike")  # Floats or integers
FloatLikeArray = TypeVar("FloatLikeArray")  # Arrays of floats or integers

NormalDistVar = TypeVar("NormalDistVar")  # Normally distributed random var
NormalDistArray = TypeVar("NormalDistArray")  # Array of Normally dist random var


def logistic_fcn(  # pylint: disable=C0103
    x: Union[FloatLikeArray, NormalDistArray],
    L: Union[FloatLike, NormalDistVar],
    k: Union[FloatLike, NormalDistVar],
    x0: Union[FloatLike, NormalDistVar],
) -> Union[FloatLikeArray, NormalDistArray]:
    """Computes `L / (1 + exp(-k(x-x0)))`.
    """
    return L / (1 + exp(-k * (x - x0)))


def one_minus_logistic_fcn(  # pylint: disable=C0103
    x: Union[FloatLikeArray, NormalDistArray],
    L: Union[FloatLike, NormalDistVar],
    k: Union[FloatLike, NormalDistVar],
    x0: Union[FloatLike, NormalDistVar],
) -> FloatLikeArray:
    """Computes `1 - L / (1 + exp(-k(x-x0)))`.
    """
    return 1 - logistic_fcn(x, L, k, x0)
