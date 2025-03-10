from . import Z, PhaseMismatch
from typing import NamedTuple, Callable
from functools import partial
import jax.numpy as jnp
from jax import lax, jit
import jax

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


def euler_step(state, dz_domain, kappa, phase_mismatch_fn):
    fund_wave_power, sh_wave_power, z = state
    dz, _ = dz_domain
    phase_mismatch_val = phase_mismatch_fn(z)

    # 微分方程式の右辺計算（kappaはすでに符号付き）
    dA_dz = -1j * jnp.conj(kappa) * jnp.conj(fund_wave_power) * \
        sh_wave_power * jnp.exp(-1j * phase_mismatch_val)
    dB_dz = -1j * kappa * fund_wave_power**2 * jnp.exp(1j * phase_mismatch_val)

    # 状態更新
    new_fund_wave_power = fund_wave_power + dz * dA_dz
    new_sh_wave_power = sh_wave_power + dz * dB_dz
    new_z = z + dz

    return (new_fund_wave_power, new_sh_wave_power, new_z), None


def integrate_domain(state, domain_tuple, kappa_magnitude, phase_mismatch_fn):
    domain_index, domain_width = domain_tuple
    # 各ドメインのindexに基づいて符号を決定（偶数: +1, 奇数: -1）
    sign = jnp.where((domain_index % 2) == 0, 1.0, -1.0)
    # 符号付きのkappaを定義
    kappa = kappa_magnitude * sign

    n_steps = 100
    dz = domain_width / n_steps
    dz_domain = (dz, domain_width)

    # ドメイン内の数値積分：kappaはこのドメイン内では一定
    final_state, _ = lax.scan(
        lambda state, _: euler_step(
            state, dz_domain, kappa, phase_mismatch_fn),
        state,
        None,
        length=n_steps
    )
    return final_state, None


def solve_ncme(params: NCMEParams) -> EffTensor:
    # 初期状態：z=0から開始
    init_state = (params.fund_power.astype(jnp.complex64),
                  params.sh_power.astype(jnp.complex64), 0.0)
    # 各ドメインのインデックスを作成
    domain_indices = jnp.arange(params.domain_widths.shape[0])
    # 各ドメインの情報をタプルにまとめる: (domain_index, domain_width)
    final_state, _ = lax.scan(
        lambda state, x: integrate_domain(
            state, x, params.kappa_magnitude, params.phase_mismatch_fn),
        init_state,
        (domain_indices, params.domain_widths)
    )
    _, final_sh_wave_power, _ = final_state

    # 変換効率の計算
    return jnp.abs(final_sh_wave_power)**2 / jnp.abs(params.fund_power)**2
