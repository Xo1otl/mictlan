import math
import numpy as np
from . import NCME, efficiency


class ChirpedGrating:
    def __init__(self, P0, kappa, L, r, T, pm_lambda=None, Lambda_0=None):
        self.P0 = P0  # P0 = |A0|^2
        self.kappa = kappa  # 結合係数
        self.L = L  # 固定長さ
        self.r = r  # チャープパラメータ（周期分極反転構造の場合は0）
        self.T = T  # 温度
        self.pm_lambda = pm_lambda  # 位相整合波長

        self.A0 = np.sqrt(P0)  # A0 = sqrt(P0)

        if Lambda_0 is not None:
            self.Lambda_0 = Lambda_0
        if pm_lambda is not None:
            N_omega = calculate_refractive_index(pm_lambda, T)
            N_2omega = calculate_refractive_index(pm_lambda / 2, T)
            self.Lambda_0 = (pm_lambda / 2) / (N_2omega - N_omega)
        if (Lambda_0 is None) == (pm_lambda is None):
            raise ValueError("pm_lambda または Lambda_0 のどちらか一方だけを指定してください。")

        self.K0 = 2 * np.pi / self.Lambda_0

    def Lambda_z(self, z):
        """z における波長を計算する"""
        return self.Lambda_0 / (1 + self.r * z)

    def calculate_efficiency(self, lambda_val):
        """与えられた波長での変換効率を計算する"""
        # 屈折率の計算
        N_omega = calculate_refractive_index(lambda_val, self.T)
        N_2omega = calculate_refractive_index(lambda_val / 2, self.T)

        # 伝搬定数の計算
        beta_omega = (2 * np.pi / lambda_val) * N_omega
        beta_2omega = (4 * np.pi / lambda_val) * N_2omega

        # twodeltaone関数の定義
        def twodeltaone(z):
            q = 1
            Phi_z = self.K0 * (z + (self.r / 2) * z**2)  # 位相項
            return beta_2omega * z - (2 * beta_omega * z + q * Phi_z)

        # NMCEの計算
        nmce = NCME(self.kappa, twodeltaone)
        B_L = nmce.solve(self.L, self.A0, B0=0)

        # 変換効率の計算
        return efficiency(B_L, self.A0)

    def log_values(self, lambda_val, *args, **kwargs):
        """特定の波長でのみログを出力する関数"""
        if np.isclose(lambda_val, 1.025, atol=1e-5):  # 1.031 µm のときだけログを出力
            print(f"\n波長 λ = {lambda_val} µm での計算結果:")
            print(*args, **kwargs)  # そのまま print に渡す


# MgO-doped stoichiometric lithium tantalate (MgO:SLT) の定数
mgodoped_slt_params = {
    "no": {  # 通常光線 (ordinary ray)
        "a": [4.508200, 0.084888, 0.195520, 1.157000, 8.251700, 0.023700],
        "b": [2.070400E-08, 1.444900E-08, 1.597800E-08, 4.768600E-06, 1.112700E-05],
    },
    "ne": {  # 異常光線 (extraordinary ray)
        "a": [4.561500, 0.084880, 0.192700, 5.583200, 8.306700, 0.021696],
        "b": [4.782000E-07, 3.091300E-08, 2.732600E-08, 1.483700E-05, 1.364700E-07],
    },
}


def calculate_refractive_index(lambda_um, T=24.5, material_params=mgodoped_slt_params["ne"]):
    """
    MgO-doped stoichiometric lithium tantalate (MgO:SLT) の実行屈折率を計算する関数

    Parameters:
        lambda_um (float): 波長 (μm) [必須]
        T (float): 温度 (℃), デフォルトは24.5
        material_params (dict): 材料の定数セット, デフォルトは mgodoped_slt_params["no"]

    Returns:
        float: 実行屈折率 N
    """
    # fの計算
    f = (T - 24.5) * (T + 24.5 + 2 * 273.16)

    # 波長の二乗
    lambda_sq = lambda_um ** 2

    # セルマイヤーの分散式
    a = material_params["a"]
    b = material_params["b"]
    n_sq = (
        a[0] + b[0] * f +
        (a[1] + b[1] * f) / (lambda_sq - (a[2] + b[2] * f) ** 2) +
        (a[3] + b[3] * f) / (lambda_sq - (a[4] + b[4] * f) ** 2) -
        a[5] * lambda_sq
    )

    # 実行屈折率Nの計算
    N = math.sqrt(n_sq)

    return N
