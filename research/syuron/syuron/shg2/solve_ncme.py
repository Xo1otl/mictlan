from typing import Tuple
from .solver import *
from .use_device import *
from jax import lax

type StepState = Tuple[FundPower, SHPower, StepIndex]


def runge_kutta_step(state: StepState, z0, dz, kappa, phase_mismatch_fn) -> Tuple[StepState, None]:
    fund_power, sh_power, index = state

    z = z0 + dz * index  # 累積加算すると誤差が蓄積するので、毎回計算する

    def derivative(A, B, z_val):
        phase_mismatch_val = phase_mismatch_fn(z_val)
        dA_dz = -1j * jnp.conj(kappa) * jnp.conj(A) * \
            B * jnp.exp(-1j * phase_mismatch_val)
        dB_dz = -1j * kappa * A**2 * jnp.exp(1j * phase_mismatch_val)
        return dA_dz, dB_dz

    k1_A, k1_B = derivative(fund_power, sh_power, z)
    k2_A, k2_B = derivative(fund_power + 0.5 * dz *
                            k1_A, sh_power + 0.5 * dz * k1_B, z + 0.5 * dz)
    k3_A, k3_B = derivative(fund_power + 0.5 * dz *
                            k2_A, sh_power + 0.5 * dz * k2_B, z + 0.5 * dz)
    k4_A, k4_B = derivative(fund_power + dz * k3_A,
                            sh_power + dz * k3_B, z + dz)

    new_fund_power = fund_power + \
        (dz / 6) * (k1_A + 2 * k2_A + 2 * k3_A + k4_A)
    new_sh_power = sh_power + \
        (dz / 6) * (k1_B + 2 * k2_B + 2 * k3_B + k4_B)
    new_index = index + 1

    return (new_fund_power, new_sh_power, new_index), None


type DomainState = Tuple[FundPower, SHPower, Z]


def integrate_domain(domain_state: DomainState, domain_info, kappa_magnitude, phase_mismatch_fn) -> Tuple[DomainState, None]:
    domain_index, domain_width = domain_info
    n_steps = 1000
    fund_power, sh_power, current_z = domain_state

    (new_fund_power, new_sh_power, _), _ = lax.cond(
        domain_width == 0,
        lambda state: (state, None),
        lambda state: lax.scan(
            lambda state, _: runge_kutta_step(
                state,
                current_z,
                domain_width / n_steps,
                kappa_magnitude *
                jnp.where((domain_index % 2) == 0, 1.0, -1.0),
                phase_mismatch_fn
            ),
            state,
            None,
            length=n_steps
        ),
        (fund_power, sh_power, 0)
    )

    return (new_fund_power, new_sh_power, current_z + domain_width), None


def solve_ncme(params: NCMEParams) -> EffTensor:
    init_state = (params.fund_power.astype(jnp.complex64),
                  params.sh_power.astype(jnp.complex64), 0.0)
    domain_indices = jnp.arange(params.domain_widths.shape[0])
    final_state, _ = lax.scan(
        lambda state, domain_info: integrate_domain(
            state, domain_info, params.kappa_magnitude, params.phase_mismatch_fn),
        init_state,
        (domain_indices, params.domain_widths)
    )
    _, final_sh_power, _ = final_state

    return jnp.abs(final_sh_power)**2 / jnp.abs(params.fund_power)**2
