from scipy.interpolate import interp1d
import math
import numpy as np
from . import PeriodicObserver, NCME, calculate_efficiency


class Grating:
    def __init__(self, P0, kappa, T):
        """
        Initialize the Grating class with common parameters.

        Parameters:
            P0 (float): Initial power |A0|^2.
            kappa (float): Coupling coefficient.
            T (float): Temperature (°C).
        """
        self.P0 = P0
        self.kappa = kappa
        self.T = T
        self.A0 = np.sqrt(P0)

    def setup_chirped_grating(self, r, L, pm_lambda=None, Lambda_0=None):
        """
        Setup for chirped grating with known transformation.

        Parameters:
            r (float): Chirp parameter.
            pm_lambda (float): Phase-matching wavelength (optional).
            Lambda_0 (float): Reference wavelength (optional).
        """
        self.r = r
        self.pm_lambda = pm_lambda
        self.L = L

        if Lambda_0 is not None:
            self.Lambda_0 = Lambda_0
        elif pm_lambda is not None:
            N_omega = calculate_refractive_index(pm_lambda, self.T)
            N_2omega = calculate_refractive_index(pm_lambda / 2, self.T)
            self.Lambda_0 = (pm_lambda / 2) / (N_2omega - N_omega)
        else:
            raise ValueError(
                "Either pm_lambda or Lambda_0 must be provided for chirped grating.")

        self.K0 = 2 * np.pi / self.Lambda_0
        # 既知のchirped gratingになる座標変換
        self.transformation = lambda z: z + (self.r / 2) * z**2

    def setup_general_grating(self, widths):
        """
        Setup for general grating with polarization reversal widths.

        Parameters:
            widths (np.array): Array of polarization reversal widths.
        """
        self.widths = widths
        self.L = np.sum(widths)
        self.Lambda_0 = widths[0]  # Reference wavelength
        n_layers = len(widths)
        self.K0 = 2 * np.pi / self.Lambda_0

        # Infer the coordinate transformation from widths
        self.observer = PeriodicObserver(n_layers, width=self.Lambda_0)
        discrete_transformation = self.observer.infer_transformation(
            self.widths)
        axis = np.arange(0, (n_layers+1) * self.Lambda_0, self.Lambda_0)
        self.transformation = interp1d(
            axis, discrete_transformation, kind='linear', fill_value='extrapolate')  # type: ignore

    def calculate_efficiency(self, lambda_val):
        """
        Calculate the conversion efficiency for a given wavelength.

        Parameters:
            lambda_val (float): Wavelength (μm).

        Returns:
            efficiency_val (float): Conversion efficiency.
        """
        # 縦型疑似位相整合では屈折率はzによらず一定
        N_omega = calculate_refractive_index(lambda_val, self.T)
        N_2omega = calculate_refractive_index(lambda_val / 2, self.T)

        # ベータも一定
        beta_omega = (2 * np.pi / lambda_val) * N_omega
        beta_2omega = (4 * np.pi / lambda_val) * N_2omega

        # 2Δ_1の式の中のΦ(z)が座標変換に対応している
        def twodeltaone(z):
            q = 1
            Phi_z = self.K0 * self.transformation(z)
            return beta_2omega * z - (2 * beta_omega * z + q * Phi_z)

        # NCME を解く
        ncme = NCME(self.kappa, twodeltaone)
        B_L = ncme.solve(self.L, self.A0, B0=0)

        # Calculate efficiency
        efficiency = calculate_efficiency(B_L, self.A0)

        return efficiency

    def debug_print(self, lambda_val, *args, **kwargs):
        """特定の波長でのみログを出力する関数"""
        if np.isclose(lambda_val, 1.025, atol=1e-5):  # 1.031 µm のときだけログを出力
            print(f"\n波長 λ = {lambda_val} µm での計算結果:")
            print(*args, **kwargs)  # そのまま print に渡す


# # MgO-doped stoichiometric lithium tantalate (MgO:SLT) の定数
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
