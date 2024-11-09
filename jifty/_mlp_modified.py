r"""
_mlp_modified.py

Implements the MLPModified class that dynamically switches between using
standard dense layers and layers with Random Weight Factorization (RWF),
based on the provided configuration.

References:
1) Wang et al. paper on RWF: https://arxiv.org/pdf/2308.08468.pdf

Usage: Refer to `example_mlps.ipynb` in the examples folder.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Union, Dict, List
import equinox as eqx

from ._layer_random_weight_factor import LinearRWF


class MLPModified(eqx.Module):
    r"""
    A modified Multilayer Perceptron (MLP) with enhanced input mixing.

    Incorporates two special encoding layers, `U` and `V`, for input mixing,
    alongside a flexible architecture that allows for either standard dense
    layers or layers equipped with Random Weight Factorization (RWF). This
    design is geared towards improving the MLP's capability in learning
    complex and non-linear solutions, particularly for Partial Differential
    Equations (PDEs).

    The model's structure, inspired by the work of Wang et al., enhances
    traditional MLP outputs by integrating the transformations:

    .. math::
        U = \sigma(W_1 x + b_1), \quad V = \sigma(W_2 x + b_2),

    and subsequently,

    .. math::
        f^{(l)}(x) = W^{(l)} \cdot g^{(l-1)}(x) + b^{(l)}, \quad
        g^{(l)}(x) = \sigma(f^{(l)}(x)) \odot U + (1 - \sigma(f^{(l)}(x))) \odot V,

    culminating in the output:

    .. math::
        f_\theta(x) = W^{(L+1)} \cdot g^{(L)}(x) + b^{(L+1)},

    where \(\odot\) denotes element-wise multiplication, thereby enriching
    the model's expressiveness and solution adaptability.

    :param in_size: The number of input features to the model.
    :type in_size: int
    :param out_size: The number of output features from the model.
    :type out_size: int
    :param width_size: The width (number of units) of the hidden layers.
    :type width_size: int
    :param depth: The total number of layers in the MLP, including the final layer.
    :type depth: int
    :param reparam_config: Configuration for weight reparameterization; influences
                           the choice between eqx.nn.Linear and LinearRWF.
                           Defaults to {"mean": 0.5, "stddev": 0.1}.
    :type reparam_config: Union[None, Dict], optional
    :param activation: The activation function applied to intermediate layer outputs.
                       Defaults to jax.nn.tanh.
    :type activation: Callable
    :param final_activation: The activation function applied to the final layer output.
                             Defaults to a linear activation (lambda x: x).
    :type final_activation: Callable
    :param key: A JAX random key used for layer weight initialization.
    :type key: jax.random.PRNGKey

    Attributes:
        u_layer (Union[eqx.nn.Linear, LinearRWF]): Encoding layer `U` for input mixing.
        v_layer (Union[eqx.nn.Linear, LinearRWF]): Encoding layer `V` for input mixing.
        layers (List[Union[eqx.nn.Linear, LinearRWF]]): List of intermediate layers.
        final_layer (Union[eqx.nn.Linear, LinearRWF]): The final output layer.
        activation (Callable): Activation function for intermediate layers.
        final_activation (Callable): Activation function for the output layer.
    """
    u_layer: Union[eqx.nn.Linear, LinearRWF]
    v_layer: Union[eqx.nn.Linear, LinearRWF]
    layers: List[Union[eqx.nn.Linear, LinearRWF]]
    final_layer: Union[eqx.nn.Linear, LinearRWF]
    activation: Callable = eqx.static_field()  # Mark as static
    final_activation: Callable = eqx.static_field()  # Mark as static
    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        reparam_config: Union[None, Dict] = {
            "mean": 0.5,
            "stddev": 0.1,
        },
        activation: Callable = jax.nn.tanh,
        final_activation: Callable = lambda x: x,
        *,
        key: jax.random.PRNGKey,
    ):
        keys = jax.random.split(key, 2 + depth)

        if reparam_config is None:
            LayerCls = lambda in_size, out_size, key, **kwargs: eqx.nn.Linear(
                in_features=in_size, out_features=out_size, use_bias=True, key=key
            )
        else:
            LayerCls = lambda in_size, out_size, key, **kwargs: LinearRWF(
                in_size=in_size, out_size=out_size, key=key, **kwargs
            )

        self.u_layer = LayerCls(in_size=in_size, out_size=width_size, key=keys[0])
        self.v_layer = LayerCls(in_size=in_size, out_size=width_size, key=keys[1])

        self.layers = [
            LayerCls(
                in_size=in_size if i == 0 else width_size,
                out_size=width_size,
                key=keys[i + 2],
            )
            for i in range(depth - 1)
        ]
        self.final_layer = LayerCls(width_size, out_size, keys[-1])

        self.activation = activation
        self.final_activation = final_activation 
    def __call__(self, x: jnp.ndarray, key=None) -> jnp.ndarray:
        """
        Forward pass through the modified MLP.

        Applies the model's layers to the input `x`, performing the enhanced input
        mixing and processing through either standard dense or RWF-equipped layers,
        as configured. The process enriches the model's expressiveness and adaptability
        in learning complex and non-linear solutions.

        :param x: The input to the MLP.
        :type x: jnp.ndarray
        :param key: An optional JAX random key for operations requiring randomness.
                    Unused in this implementation but included for API compatibility.
        :type key: jax.random.PRNGKey, optional
        :return: The output after the forward pass through the MLP.
        :rtype: jnp.ndarray
        """
        u = self.activation(self.u_layer(x))
        v = self.activation(self.v_layer(x))
        for layer in self.layers:
            x = self.activation(layer(x))
            x = x * u + (1 - x) * v
        x = self.final_layer(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x
