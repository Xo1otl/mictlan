from typing import Callable
import jax.numpy as jnp

type Z = float
type PhaseMismatch = jnp.ndarray
type Wavelength = jnp.ndarray
type T = jnp.ndarray

type PhaseMismatchFn = Callable[[Z], PhaseMismatch]
type UseDevice = Callable[[Wavelength, T], PhaseMismatchFn]
