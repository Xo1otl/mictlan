from .device import *
from typing import NamedTuple, Callable
import jax.numpy as jnp

type EffTensor = jnp.ndarray
type FundPower = jnp.ndarray
type SHPower = jnp.ndarray
type KappaMagnitude = jnp.ndarray
type DomainWidths = jnp.ndarray
type StepIndex = int


class NCMEParams(NamedTuple):
    fund_power: FundPower
    sh_power: SHPower
    kappa_magnitude: KappaMagnitude
    phase_mismatch_fn: PhaseMismatchFn
    domain_widths: DomainWidths


type SolverFn = Callable[[NCMEParams], EffTensor]
