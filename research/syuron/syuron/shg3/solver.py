from .use_material import *
from typing import NamedTuple, Callable, List
import jax.numpy as jnp

type FundPower = jnp.ndarray
type SHPower = jnp.ndarray
type KappaMagnitude = jnp.ndarray
type EffTensor = jnp.ndarray


class Domain(NamedTuple):
    width: float
    kappa: float


type DomainStack = List[Domain]


class NCMEParams(NamedTuple):
    fund_power: FundPower
    sh_power: SHPower
    domain_stack: DomainStack
    phase_mismatch_fn: PhaseMismatchFn
    mesh_density: int


type SolverFn = Callable[[NCMEParams], EffTensor]
