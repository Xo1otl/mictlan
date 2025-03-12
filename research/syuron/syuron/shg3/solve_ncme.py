from typing import Tuple
from .solver import *
from .use_material import *
from jax import lax

type StepIndex = int
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


type DomainIndex = int
type DomainState = Tuple[FundPower, SHPower, Z, DomainIndex]


def integrate_domain(state: DomainState, domain_stack: DomainStack, phase_mismatch_fn: PhaseMismatchFn, mesh_density: int) -> Tuple[DomainState, None]:
    fund_power, sh_power, current_z, domain_index = state
    domain_width, kappa = domain_stack[domain_index]

    (new_fund_power, new_sh_power, _), _ = lax.cond(
        domain_width == 0,
        lambda state: (state, None),
        lambda state: lax.scan(
            lambda state, _: runge_kutta_step(
                state,
                current_z,
                domain_width / mesh_density,
                kappa,
                phase_mismatch_fn
            ),
            state,
            None,
            length=mesh_density
        ),
        (fund_power, sh_power, 0)
    )

    return (new_fund_power, new_sh_power, current_z + domain_width, domain_index + 1), None


def solve_ncme(params: NCMEParams) -> EffTensor:
    init_state = (params.fund_power.astype(jnp.complex64),
                  params.sh_power.astype(jnp.complex64), 0.0, 0)
    final_state, _ = lax.scan(
        lambda state, _: integrate_domain(
            state, params.domain_stack, params.phase_mismatch_fn, mesh_density=params.mesh_density),
        init_state,
        length=len(params.domain_stack)
    )
    _, final_sh_power, _, _ = final_state

    return jnp.abs(final_sh_power)**2 / jnp.abs(params.fund_power)**2
