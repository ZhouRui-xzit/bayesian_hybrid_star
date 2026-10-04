"""JAX float64 kernels for compact-star calculations."""

from jax import config

config.update("jax_enable_x64", True)
