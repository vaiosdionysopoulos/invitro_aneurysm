r"""
_layer_fourier.py

Implements the FourierFeaturesLayer class for Random Fourier Feature mapping to
enhance machine learning models' capability to learn high-frequency patterns.
This addresses the spectral bias of MLPs in Physics-Informed Neural Networks
(PINNs).

References:
1) Tancik et al. 2020: https://arxiv.org/abs/2006.10739

Usage:
- This class is intended for use in constructing neural network models where
  fourier features mapping is required to transform input data into a
  high-dimensional space.
"""

__all__ = ["FourierFeaturesLayer"]

import jax
import jax.numpy as jnp
import equinox as eqx


class FourierFeaturesLayer(eqx.Module):
    r"""
    Implements Random Fourier Feature mapping to transform input data into a
    high-dimensional space, mitigating MLPs' spectral bias and enhancing PINNs'
    learning of high-frequency patterns and complex solutions.

    The mapping follows:

    .. math::
        \gamma(\mathbf{x}) = \begin{bmatrix}
            \sin(\mathbf{B}\mathbf{x}) \\
            \cos(\mathbf{B}\mathbf{x})
        \end{bmatrix},

    where :math:`\mathbf{B} \in \mathbb{R}^{m \times d}` is sampled from
    :math:`\mathcal{N}(0, \text{scale}^2)`, 'd' is the input dimension, and 'm'
    is the number of Fourier features, transforming input from :math:`\mathbb{R}^n`
    to :math:`\mathbb{R}^{2m}` by concatenating sine and cosine transformations.

    :param in_size: Dimension of input space (d).
    :type in_size: int
    :param num_fourier_features: Number of Fourier features (m) in output mapping,
                                 leading to 2m output dimension.
    :type num_fourier_features: int
    :param key: JAX random key for generating projections.
    :type key: jax.random.PRNGKey
    :param scale: Scale factor for projection matrix, affecting frequency domain of
                  embedding. Recommended range is [1, 10], defaulting to 5.0.
    :type scale: float

    Attributes:
        B (jax.numpy.ndarray): Projection matrix for random Fourier features, shaped
                               (in_size, num_fourier_features), scaled by `scale`.
    """

    B: jax.Array

    def __init__(
        self,
        in_size: int,
        num_fourier_features: int,
        key: jax.random.PRNGKey,
        scale: float = 5.0,
    ):
        """
        Initializes FourierFeaturesLayer with a scaled random projection matrix
        B, enhancing the input's representation with high-frequency components.

        :param in_size: Dimension of input space.
        :param num_fourier_features: Number of Fourier features for the output mapping.
        :param key: JAX random key for generating projections.
        :param scale: Scale factor for the projection matrix.
        """
        self.B = jax.random.normal(key, (in_size, num_fourier_features)) * scale

    @property
    def in_size(self) -> int:
        """Returns the dimension of the input space."""
        return self.B.shape[0]

    @property
    def out_size(self) -> int:
        """
        Returns the dimension of the output space, twice the number of Fourier
        features due to concatenation of sine and cosine values.
        """
        return self.B.shape[1] * 2

    @property
    def num_fourier_features(self) -> int:
        """Returns the number of Fourier features."""
        return self.B.shape[1]

    def __call__(self, x: jnp.ndarray, key=None) -> jnp.ndarray:
        """
        Applies Fourier feature mapping to input data, enhancing its representation
        with high-frequency components.

        :param x: Input data to be transformed.
        :type x: jnp.ndarray
        :param key: Unused in this method, but accepted for API compatibility.
        :type key: jax.random.PRNGKey, optional

        :return: Transformed input data via Fourier feature mapping, concatenating
                 sine and cosine transformations.
        :rtype: jnp.ndarray
        """
        y = x @ self.B
        return jnp.concatenate([jnp.sin(y), jnp.cos(y)], axis=-1)
