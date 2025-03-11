from . import Z, PhaseMismatch
from typing import NamedTuple, Callable, Tuple
import jax.numpy as jnp
from jax import lax

EffTensor = jnp.ndarray
FundPower = jnp.ndarray
SHPower = jnp.ndarray
KappaMagnitude = jnp.ndarray
DomainWidths = jnp.ndarray


class NCMEParams(NamedTuple):
    fund_power: FundPower
    sh_power: SHPower
    kappa_magnitude: KappaMagnitude
    phase_mismatch_fn: Callable[[Z], PhaseMismatch]
    domain_widths: DomainWidths


State = Tuple[FundPower, SHPower, Z]


def runge_kutta_step(state: State, dz, kappa, phase_mismatch_fn):
    fund_wave_power, sh_wave_power, z = state

    def derivative(A, B, z_val):
        phase_mismatch_val = phase_mismatch_fn(z_val)
        dA_dz = -1j * jnp.conj(kappa) * jnp.conj(A) * \
            B * jnp.exp(-1j * phase_mismatch_val)
        dB_dz = -1j * kappa * A**2 * jnp.exp(1j * phase_mismatch_val)
        return dA_dz, dB_dz

    k1_A, k1_B = derivative(fund_wave_power, sh_wave_power, z)
    k2_A, k2_B = derivative(fund_wave_power + 0.5 * dz *
                            k1_A, sh_wave_power + 0.5 * dz * k1_B, z + 0.5 * dz)
    k3_A, k3_B = derivative(fund_wave_power + 0.5 * dz *
                            k2_A, sh_wave_power + 0.5 * dz * k2_B, z + 0.5 * dz)
    k4_A, k4_B = derivative(fund_wave_power + dz * k3_A,
                            sh_wave_power + dz * k3_B, z + dz)

    new_fund_wave_power = fund_wave_power + \
        (dz / 6) * (k1_A + 2 * k2_A + 2 * k3_A + k4_A)
    new_sh_wave_power = sh_wave_power + \
        (dz / 6) * (k1_B + 2 * k2_B + 2 * k3_B + k4_B)
    new_z = z + dz

    return (new_fund_wave_power, new_sh_wave_power, new_z), None


def integrate_domain(state: State, domain_tuple, kappa_magnitude, phase_mismatch_fn):
    domain_index, domain_width = domain_tuple
    n_steps = 1000

    return lax.cond(
        domain_width == 0,
        lambda state: (state, None),
        lambda state: lax.scan(
            lambda state, _: runge_kutta_step(
                state,
                domain_width / n_steps,
                kappa_magnitude *
                jnp.where((domain_index % 2) == 0, 1.0, -1.0),
                phase_mismatch_fn
            ),
            state,
            None,
            length=n_steps
        ),
        state
    )


def solve_ncme(params: NCMEParams) -> EffTensor:
    init_state = (params.fund_power.astype(jnp.complex64),
                  params.sh_power.astype(jnp.complex64), 0.0)
    domain_indices = jnp.arange(params.domain_widths.shape[0])
    final_state, _ = lax.scan(
        lambda state, z: integrate_domain(
            state, z, params.kappa_magnitude, params.phase_mismatch_fn),
        init_state,
        (domain_indices, params.domain_widths)
    )
    _, final_sh_wave_power, _ = final_state

    return jnp.abs(final_sh_wave_power)**2 / jnp.abs(params.fund_power)**2
