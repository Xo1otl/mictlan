import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, List


class NCME:
    def __init__(self, kappa: complex, twodeltaone: Callable[[float], float]):
        """
        非線形媒質変換効率（NMCE）クラス。第二高調波発生（SHG）を計算する。

        パラメータ:
        kappa (complex): 非線形結合定数。
        twodeltaone (Callable[[float], float]): 2*Δ(z) を表す関数。
        """
        self.kappa = kappa
        self.twodeltaone = twodeltaone

    def solve(self, L: float, A0: complex, B0: complex) -> complex:
        """
        連立微分方程式を解き、B(L) を計算する。

        パラメータ:
        L (float): 積分範囲。
        A0 (complex): A(0) の初期条件。
        B0 (complex): B(0) の初期条件。

        戻り値:
        complex: B(L) の値。
        """
        # 微分方程式の右辺を定義
        def equations(z: float, y: List[float]) -> List[float]:
            A_real, A_imag, B_real, B_imag = y[0], y[1], y[2], y[3]
            A = A_real + 1j * A_imag
            B = B_real + 1j * B_imag
            phase_term_A = np.exp(-1j * self.twodeltaone(z))  # 位相項 A
            phase_term_B = np.exp(1j * self.twodeltaone(z))   # 位相項 B
            dA_dz = -1j * np.conj(self.kappa) * \
                np.conj(A) * B * phase_term_A  # dA/dz
            dB_dz = -1j * self.kappa * A**2 * phase_term_B  # dB/dz
            return [dA_dz.real, dA_dz.imag, dB_dz.real, dB_dz.imag]

        # 初期条件を実数部と虚数部に分ける
        y0 = [A0.real, A0.imag, B0.real, B0.imag]

        # solve_ivp で微分方程式を解く
        sol = solve_ivp(equations, [0, L], y0, method='RK45')

        # B(L) を取得
        B_L_real = sol.y[2][-1]  # z = L における B の実数部
        B_L_imag = sol.y[3][-1]  # z = L における B の虚数部
        B_L = B_L_real + 1j * B_L_imag  # 複素数に戻す
        return B_L


def calculate_efficiency(B_L, A0):
    """
    SHG効率を計算する。

    パラメータ:
    B_L (complex): z = L における第二高調波の振幅。
    A0 (complex): 基本波の初期振幅。

    戻り値:
    float: SHG効率。
    """
    return abs(B_L)**2 / abs(A0)**2
