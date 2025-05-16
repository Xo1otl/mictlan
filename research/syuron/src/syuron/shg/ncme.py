from .domain import *
from .use_material import *
from typing import NamedTuple, Callable
import jax.numpy as jnp

type FundPower = jnp.ndarray
type SHPower = jnp.ndarray
type KappaMagnitude = jnp.ndarray
type EffTensor = jnp.ndarray


class NCMEParams(NamedTuple):
    fund_power: FundPower
    sh_power: SHPower
    grating: Grating
    phase_mismatch_fn: PhaseMismatchFn
    mesh_density: int


type NCMESolverFn = Callable[[NCMEParams], EffTensor]
