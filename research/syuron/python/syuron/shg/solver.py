from . import Device
from typing import Protocol
import jax.numpy as jnp
from jax import lax


class NCMESolver(Protocol):
    def solve(self, A0: complex, B0: complex = 0) -> jnp.ndarray:
        """波長変換効率の計算

        Nonlinear Coupled Mode Equations を解いて波長変換効率を計算する。

        Args:
            A0 (complex): 基本波 A(0) の初期条件
            B0 (complex): 第二高調波 B(0) の初期条件、デフォルトは 0
        """
        ...


class EulerNCMESolver(NCMESolver):
    def __init__(self, device: Device, wavelength: jnp.ndarray, T: jnp.ndarray):
        self.kappa = device.kappa
        self.phase_mismatch = device.phase_mismatch(wavelength, T)
        self.z_mesh = device.z_mesh

    def solve(self, A0: complex, B0: complex = 0) -> jnp.ndarray:
        # 初期条件
        A0_arr = jnp.full_like(self.phase_mismatch(0.0),
                               A0, dtype=jnp.complex64)
        B0_arr = jnp.full_like(self.phase_mismatch(0.0),
                               B0, dtype=jnp.complex64)
        init_state = (A0_arr, B0_arr)

        # z_meshから得られる(z, dz)ペアごとに実行する関数
        def euler_step(state, z_dz):
            A, B = state
            z, dz = z_dz

            # 現在のzにおけるパラメータを取得
            kappa_val = self.kappa(z)
            phase_mismatch_val = self.phase_mismatch(z)

            # 微分方程式の右辺を計算
            dA_dz = -1j * jnp.conj(kappa_val) * jnp.conj(A) * \
                B * jnp.exp(-1j * phase_mismatch_val)
            dB_dz = -1j * kappa_val * A**2 * jnp.exp(1j * phase_mismatch_val)

            # 前進オイラー法による更新
            new_A = A + dz * dA_dz
            new_B = B + dz * dB_dz

            return (new_A, new_B), None

        final_state, _ = lax.scan(euler_step, init_state, self.z_mesh())

        A_L, B_L = final_state

        return self._calc_eff(B_L, A0)

    def _calc_eff(self, B_L, A0) -> jnp.ndarray:
        return abs(B_L)**2 / abs(A0)**2
