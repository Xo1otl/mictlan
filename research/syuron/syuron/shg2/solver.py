from . import Z, PhaseMismatch
from typing import NamedTuple, Callable
import jax.numpy as jnp
from jax import lax, jit

Eff = jnp.ndarray
FundWavePower = jnp.ndarray
SHWavePower = jnp.ndarray
KappaMagnitude = jnp.ndarray
Widths = jnp.ndarray


class NCMEParams(NamedTuple):
    fund_wave_power: FundWavePower
    sh_wave_power: SHWavePower
    kappa_magnitude: KappaMagnitude
    phase_mismatch_fn: Callable[[Z], PhaseMismatch]
    widths: Widths


def integrate(state, x):
    A, B = state
    return (A, B), None


def integrate_widths(carry, x):
    A, B = carry
    init_state = (A, B)
    lax.scan(integrate, init_state, None, 1000)  # 各幅で1000分割して積分したものを逐次的に足していく
    return carry, None


def solve_ncme(params: NCMEParams) -> Eff:
    init_state = (params.fund_wave_power, params.sh_wave_power)
    lax.scan(integrate_widths, init_state, params.widths)
    raise NotImplementedError
