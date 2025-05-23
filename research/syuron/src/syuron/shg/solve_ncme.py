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

    # FIXME: ここをルンゲクッタ法ではなく、解析的に解く方法があるらしい
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


# FIXME: z以外で前後関係が不要なため、逐次計算する必要がない
#  z_globalだけ先にcumsum的なもので計算しておいて、この関数は並列実行、最後に総和をとる方が計算が速そう
def integrate_domain_npda(state: DomainState,
                          domain: Domain,
                          # Assumes this returns Phi(z)
                          phase_mismatch_fn: PhaseMismatchFn,
                          ) -> Tuple[DomainState, None]:
    fund_power_in, sh_power_in, current_z_global = state
    domain_width, kappa_d = domain

    fund_power_out = fund_power_in
    A_in_sq = fund_power_in**2
    Ld = domain_width

    phi_start_global = phase_mismatch_fn(current_z_global)
    phi_end_global = phase_mismatch_fn(current_z_global + Ld)

    alpha = (phi_end_global - phi_start_global) / 2.0

    sinc_argument = alpha / jnp.pi
    sinc_val = jnp.sinc(sinc_argument)

    exp_term_local_sinc = jnp.exp(1j * alpha)

    generated_sh_component_relative_phase = -1j * kappa_d * \
        A_in_sq * Ld * exp_term_local_sinc * sinc_val

    global_phase_factor_at_domain_start = jnp.exp(1j * phi_start_global)
    generated_sh_component = generated_sh_component_relative_phase * \
        global_phase_factor_at_domain_start

    sh_power_out = sh_power_in + generated_sh_component
    new_z_global = current_z_global + domain_width

    return (fund_power_out, sh_power_out, new_z_global), None


def solve_ncme(params: NCMEParams) -> EffTensor:
    init_state = (params.fund_power.astype(jnp.complex64),
                  params.sh_power.astype(jnp.complex64), 0.0)
    # scan_fn = partial(integrate_domain, phase_mismatch_fn=params.phase_mismatch_fn,
    #                   mesh_density=params.mesh_density)
    scan_fn = partial(integrate_domain_npda,
                      phase_mismatch_fn=params.phase_mismatch_fn)
    final_state, _ = lax.scan(
        scan_fn,
        init_state,
        xs=params.grating  # type: ignore pylanceでエラーが出るけど無視したら動く、推論のバグ？
    )
    _, final_sh_power, _ = final_state

    return final_sh_power / params.fund_power


# FIXME: 逐次計算する必要がない部分を並列化することでさらに高速化が可能な予定だけど、今の計算でもコンパイル後は1秒もかからないので、オプティマイザと組み合わせても行ける
# def solve_ncme_npda_parallel(params: NCMEParams) -> EffTensor:
#     # TODO: z以外で前後関係が不要なため、逐次計算する必要がない
#     #  z_globalだけ先にcumsum的なもので計算しておいて、この関数は並列実行、最後に総和をとる方が計算が速そう
#     init_state = (params.fund_power.astype(jnp.complex64),
#                   params.sh_power.astype(jnp.complex64), 0.0)

#     # TODO: 各ドメインの先頭位置のz_global配列をあらかじめ計算する
#     # TODO: z_globalとparams.gratingの各要素であるdomain(domain_width, kappa_dのタプル)準備
#     # TODO: z_globalの配列とdomainの配列がある、同じ長さである、これらに対してintegrate_domain_npdaを並列実行する
#     # TODO: 計算結果の和をとる (generated_sh_componentの和に対応するはず)
#     final_sh_power = None
#     return final_sh_power / params.fund_power

# def solve_ncme_npda_parallel(params: NCMEParams) -> EffTensor:
#     """
#     NDPAを用いてNCMEを並列に解きます。

#     Args:
#         params: NCMEの計算に必要なパラメータ (NCMEParams)。
#                 - fund_power: 基本波パワー (複素振幅)
#                 - sh_power: 初期SHパワー (複素振幅)
#                 - grating: ドメイン構造 [(幅1, κ1), (幅2, κ2), ...]
#                 - phase_mismatch_fn: 位相不整合関数 Phi(z)

#     Returns:
#         効率テンソル (最終SHパワー / 基本波パワー)。
#     """
#     # パラメータの取得
#     fund_power = params.fund_power.astype(jnp.complex64)
#     sh_power = params.sh_power.astype(jnp.complex64)
#     grating = params.grating
#     phase_mismatch_fn = params.phase_mismatch_fn

#     # gratingがJAX配列でない場合、変換する
#     if not isinstance(grating, jnp.ndarray):
#         grating_array = jnp.array(grating)
#     else:
#         grating_array = grating

#     # ドメイン幅を取得
#     domain_widths = grating_array[:, 0]

#     # 各ドメインの開始位置 z_global を計算
#     # jnp.cumsum を使用して累積和を計算し、先頭に 0.0 を追加
#     z_starts = jnp.concatenate(
#         (jnp.array([0.0]), jnp.cumsum(domain_widths[:-1])))

#     def calculate_sh_component_single(
#         current_z_global: Z,
#         domain: Domain,
#         fund_p: FundPower,
#         pm_fn: PhaseMismatchFn
#     ) -> SHPower:
#         """単一ドメインで生成されるSH成分を計算します (NDPA)。"""
#         domain_width, kappa_d = domain
#         A_in_sq = fund_p**2
#         Ld = domain_width

#         phi_start_global = pm_fn(current_z_global)
#         phi_end_global = pm_fn(current_z_global + Ld)

#         alpha = (phi_end_global - phi_start_global) / 2.0

#         # jnp.sinc(x) = sin(pi*x)/(pi*x) を利用
#         sinc_argument = alpha / jnp.pi
#         sinc_val = jnp.sinc(sinc_argument)

#         exp_term_local_sinc = jnp.exp(1j * alpha)

#         # -1j * kappa * A^2 * Ld * exp(1j * (phi_start + phi_end)/2) * sinc(alpha/pi)
#         generated_sh_component_relative_phase = -1j * kappa_d * \
#             A_in_sq * Ld * exp_term_local_sinc * sinc_val

#         global_phase_factor_at_domain_start = jnp.exp(1j * phi_start_global)
#         generated_sh_component = generated_sh_component_relative_phase * \
#             global_phase_factor_at_domain_start

#         return generated_sh_component

#     # jax.vmap を使って関数をベクトル化 (並列化)
#     # in_axes は、どの引数をマッピングするかを指定します。
#     # (0, 0, None, None) は、z_starts と grating_array の最初の軸に沿ってマッピングし、
#     # fund_power と phase_mismatch_fn はブロードキャスト (そのまま使用) します。
#     vmapped_calculator = jax.vmap(
#         calculate_sh_component_single,
#         in_axes=(0, 0, None, None)
#     )

#     # 全てのドメインについてSH成分を並列計算
#     all_sh_components = vmapped_calculator(
#         z_starts,
#         grating_array,
#         fund_power,
#         phase_mismatch_fn
#     )

#     # 全ての成分を合計し、初期SHパワーを加算
#     final_sh_power = sh_power + jnp.sum(all_sh_components)

#     # 効率 (振幅比) を計算して返す
#     # ゼロ除算を避ける (fund_power が 0 の場合は 0 を返す)
#     return jnp.where(
#         fund_power == 0,
#         0.0 + 0.0j,  # 複素数で返す
#         final_sh_power / fund_power
#     )
