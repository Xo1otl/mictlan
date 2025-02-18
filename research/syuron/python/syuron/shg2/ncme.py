import math
import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, List


class NCME:
    def __init__(self, kappa: Callable[[float], complex], phase_mismatch: Callable[[float], float]):
        """
        非線形媒質変換効率（NMCE）クラス。第二高調波発生（SHG）を計算する。

        パラメータ:
        kappa (Callable[[float], float]): 位置zにおける非線形結合定数を返す関数。
        phase_mismatch (Callable[[float], float]): 位相不整合 2*Δ(z) を表す関数。
        """
        self.kappa = kappa
        self.phase_mismatch = phase_mismatch

    def solve(self, L: float, A0: complex, B0: complex) -> float:
        """
        連立微分方程式を前進オイラー法で解き、変換効率を計算する。

        パラメータ:
          L (float): 積分範囲。
          A0 (complex): A(0) の初期条件。
          B0 (complex): B(0) の初期条件。

        戻り値:
          float: 変換効率。
        """
        # 初期条件を実数部と虚数部に分ける
        y = np.array([A0.real, A0.imag, B0.real, B0.imag], dtype=float)

        # 積分ステップ数を設定（必要に応じて調整してください）
        steps = 10000
        h = L / steps  # ステップ幅
        z = 0.0

        # 前進オイラー法による積分
        for _ in range(steps):
            # 現在の状態から A, B を復元
            A = y[0] + 1j * y[1]
            B = y[2] + 1j * y[3]

            # 現在の z におけるパラメータを取得
            kappa_val = self.kappa(z)
            phase_mismatch_val = self.phase_mismatch(z)

            # 微分方程式の右辺を計算
            dA_dz = -1j * np.conj(kappa_val) * np.conj(A) * \
                B * np.exp(-1j * phase_mismatch_val)
            dB_dz = -1j * kappa_val * A**2 * np.exp(1j * phase_mismatch_val)

            # 微分値の実数部・虚数部を配列にまとめる
            dy = np.array([dA_dz.real, dA_dz.imag, dB_dz.real,
                          dB_dz.imag], dtype=float)

            # 前進オイラー法による更新
            y += h * dy
            z += h

        # z = L における B の値を取得
        B_L = y[2] + 1j * y[3]
        return self._calc_eff(B_L, A0)

    def _calc_eff(self, B_L, A0):
        return abs(B_L)**2 / abs(A0)**2
