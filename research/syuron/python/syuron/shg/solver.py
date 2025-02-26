import numpy as np
from . import Device
from typing import Protocol


class NCMESolver(Protocol):
    def solve(self, A0: complex, B0: complex = 0) -> float:
        """波長変換効率の計算

        SHG の波長変換効率を計算する。

        Args:
            L (float): 波長変換が行われる領域の長さ
            A0 (complex): 基本波 A(0) の初期条件
            B0 (complex): 第二高調波 B(0) の初期条件、デフォルトは 0
        """
        ...


class EulerNCMESolver(NCMESolver):
    def __init__(self, device: Device, wavelength: float, T: float):
        self.kappa = device.kappa
        self.phase_mismatch = device.phase_mismatch(wavelength, T)
        self.z_mesh = device.z_mesh

    def solve(self, A0: complex, B0: complex = 0) -> float:
        # 初期条件
        A, B = A0, B0

        # 前進オイラー法による積分
        for z, dz in self.z_mesh():
            # 現在の z におけるパラメータを取得
            kappa_val = self.kappa(z)
            phase_mismatch_val = self.phase_mismatch(z)

            # 微分方程式の右辺を計算
            dA_dz = -1j * np.conj(kappa_val) * np.conj(A) * \
                B * np.exp(-1j * phase_mismatch_val)
            dB_dz = -1j * kappa_val * A**2 * np.exp(1j * phase_mismatch_val)

            # 前進オイラー法による更新
            A += dz * dA_dz
            B += dz * dB_dz

        return self._calc_eff(B, A0)

    def _calc_eff(self, B_L, A0):
        return abs(B_L)**2 / abs(A0)**2
