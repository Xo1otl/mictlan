import jax.numpy as jnp
from typing import Iterable, Tuple, NamedTuple

type Inputs = jnp.ndarray
type Outputs = jnp.ndarray
type Dataset = Iterable[Tuple[Inputs, Outputs]]


class Batch(NamedTuple):
    inputs: jnp.ndarray
    outputs: jnp.ndarray


type BatchSize = int
