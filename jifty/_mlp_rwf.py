r"""
_mlp_rwf.py

Implements the MLPRWF function that constructs a Multilayer Perceptron (MLP)
utilizing Random Weight Factorization (RWF).

References:
1) Wang et al. paper: https://arxiv.org/pdf/2308.08468.pdf

Usage: Refer to `example_mlps.ipynb` in the examples folder.
"""


__all__ = ["MLPRWF"]

import jax
import equinox as eqx
from typing import Callable

from ._layer_random_weight_factor import LinearRWF


def MLPRWF(
    in_size: int,
    out_size: int,
    width_size: int,
    depth: int,
    activation: Callable = jax.nn.tanh,
    final_activation: Callable = lambda x: x,
    mean: float = 0.5,
    stddev: float = 0.1,
    *,
    key: jax.random.PRNGKey,
):
    r"""
    Constructs a Multilayer Perceptron (MLP) utilizing Random Weight
    Factorization (RWF) to enhance training performance and model accuracy.

    This function creates an MLP that incorporates RWF in each of its layers,
    aiming to improve parameter efficiency and potentially lead to better
    generalization.

    :param in_size: The number of input features.
    :type in_size: int
    :param out_size: The number of output features.
    :type out_size: int
    :param width_size: The width of the hidden layers.
    :type width_size: int
    :param depth: The total number of layers, including input and output.
    :type depth: int
    :param activation: The activation function for hidden layers,
        defaults to jax.nn.tanh.
    :type activation: Callable
    :param final_activation: The activation function for the output layer,
        defaults to a linear activation (lambda x: x).
    :type final_activation: Callable
    :param mean: The mean for the normal distribution used in RWF,
        defaults to 0.5.
    :type mean: float
    :param stddev: The standard deviation for the distribution,
        defaults to 0.1.
    :type stddev: float
    :param key: The key for random initialization.
    :type key: jax.random.PRNGKey
    :return: A sequential model comprising LinearRWF layers and activation
        functions.
    :rtype: eqx.nn.Sequential
    """
    keys = jax.random.split(key, depth)
    layers = []
    for i in range(depth):
        layers.append(
            LinearRWF(
                in_size=in_size if i == 0 else width_size,
                out_size=width_size if i < depth - 1 else out_size,
                key=keys[i],
                mean=mean,
                stddev=stddev,
            )
        )
        if i < depth - 1:
            layers.append(eqx.nn.Lambda(activation))
    if final_activation is not None:
        layers.append(eqx.nn.Lambda(final_activation))

    return eqx.nn.Sequential(layers)
