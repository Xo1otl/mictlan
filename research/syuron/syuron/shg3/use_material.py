from typing import Callable
import jax.numpy as jnp

type Z = float  # z に対して微分方程式を解く、これだけ並列化が不可能なので float
type PhaseMismatch = jnp.ndarray
type Wavelength = jnp.ndarray
type T = jnp.ndarray

type PhaseMismatchFn = Callable[[Z], PhaseMismatch]
type UseMaterial = Callable[[Wavelength, T], PhaseMismatchFn]
