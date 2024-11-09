r"""
_layer_random_weight_factor.py

Implements the LinearRWF class, utilizing Random Weight Factorization (RWF) to
enhance the performance and accuracy of neural networks by factorizing
weights into scale factors and base vectors.

References:
1) Wang et al. paper: https://arxiv.org/pdf/2308.08468.pdf
2) jaxpi GitHub repository: https://github.com/PredictiveIntelligenceLab/jaxpi

Usage:
- This class is intended for use in constructing neural network models where
  dense layers with Random Weight Factorization (RWF) are required.
"""


__all__ = ["LinearRWF"]

import jax
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal, zeros
from typing import Callable
import equinox as eqx


class LinearRWF(eqx.Module):
    r"""Implements a layer using Random Weight Factorization (RWF).

    RWF improves neural network performance by decomposing each weight into
    the product of a scale factor and a base vector. This approach aims to
    enhance training dynamics and model accuracy. The decomposition is given by
    \(w^{(i,j)} = g^{(j)} \cdot v^{(i,j)}\), where \(w^{(i,j)}\) is the weight
    from the \(i\)-th input feature to the \(j\)-th output feature, \(g^{(j)}\)
    is the scale factor for the \(j\)-th output, and \(v^{(i,j)}\) is the base
    vector component of the weight.

    :param in_size: The number of input features.
    :type in_size: int
    :param out_size: The number of output features.
    :type out_size: int
    :param kernel_init: Initializer for the weight matrix, defaults to Glorot normal.
    :type kernel_init: Callable, optional
    :param bias_init: Initializer for the bias vector, defaults to zeros.
    :type bias_init: Callable, optional
    :param mean: Mean for the normal distribution of log-scale factors, defaults to 0.5.
    :type mean: float, optional
    :param stddev: Standard deviation for the distribution, defaults to 0.1.
    :type stddev: float, optional
    :param key: The key for random number generation.
    :type key: jax.random.PRNGKey

    Attributes:
        g (jnp.ndarray): The scale factors for weight factorization.
        v (jnp.ndarray): The base vectors for weight factorization.
        bias (jnp.ndarray): The bias vector for the layer.
    """

    g: jnp.ndarray
    v: jnp.ndarray
    bias: jnp.ndarray

    def __init__(
        self,
        in_size: int,
        out_size: int,
        kernel_init: Callable = glorot_normal(),
        bias_init: Callable = zeros,
        mean: float = 0.5,
        stddev: float = 0.1,
        *,
        key: jax.random.PRNGKey,
    ):
        key_g, key_v, key_bias = jax.random.split(key, 3)
        w = kernel_init(key_v, (in_size, out_size))
        g = mean + jax.random.normal(key_g, (out_size,)) * stddev
        g = jnp.exp(g)
        self.g = g
        self.v = w / g
        self.bias = bias_init(key_bias, (out_size,))

    def __call__(self, x: jnp.ndarray, key=None) -> jnp.ndarray:
        """
        Applies the layer to the input, performing weight factorization and bias addition.

        :param x: The input array.
        :type x: jnp.ndarray
        :param key: Unused, but accepted for API compatibility.
        :type key: jax.random.PRNGKey, optional
        :return: The output after applying the weight factorization and bias.
        :rtype: jnp.ndarray
        """
        kernel = self.g * self.v
        return jnp.dot(x, kernel) + self.bias
