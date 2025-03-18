from typing import Tuple
from .domain import *
from .ncme import *
from .analyze import *
from .use_material import *
from jax import lax
from functools import partial

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
type DomainState = Tuple[FundPower, SHPower, Z]


def integrate_domain(state: DomainState, domain: Domain, phase_mismatch_fn: PhaseMismatchFn, mesh_density: int) -> Tuple[DomainState, None]:
    fund_power, sh_power, current_z = state
    domain_width, kappa = domain

    scan_fn = partial(
        runge_kutta_step,
        z0=current_z,
        dz=domain_width / mesh_density,
        kappa=kappa,
        phase_mismatch_fn=phase_mismatch_fn
    )

    (new_fund_power, new_sh_power, _), _ = lax.scan(
        lambda state, _: scan_fn(state),
        (fund_power, sh_power, 0),
        length=mesh_density
    )

    return (new_fund_power, new_sh_power, current_z + domain_width), None


def solve_ncme(params: NCMEParams) -> EffTensor:
    init_state = (params.fund_power.astype(jnp.complex64),
                  params.sh_power.astype(jnp.complex64), 0.0)
    scan_fn = partial(integrate_domain, phase_mismatch_fn=params.phase_mismatch_fn,
                      mesh_density=params.mesh_density)
    final_state, _ = lax.scan(
        scan_fn,
        init_state,
        xs=params.superlattice  # type: ignore pylanceでエラーが出るけど無視したら動く、推論のバグ？
    )
    _, final_sh_power, _ = final_state

    return final_sh_power / params.fund_power
