from typing import Callable
import jax.numpy as jnp
from jax import jit


Z = jnp.float16
PhaseMismatch = jnp.ndarray
Wavelength = jnp.ndarray
T = jnp.ndarray


def usePPMgOSLT(wavelength: Wavelength, t: T) -> Callable[[Z], PhaseMismatch]:
    """
    周期分極構造の幅リストからPPMgOSLTのパラメータを計算する関数

    Args:
        widths: 分極ドメインの幅リスト (m)
        kappa_magnitude: 非線形結合係数の大きさ

    Returns:
        kappa_magnitude: 非線形結合係数の大きさ (constant)
        phase_mismatch: 波長と温度に依存する位相不整合関数
    """
    # 材料パラメータ
    params = {
        "no": {  # 通常光線 (ordinary ray)
            "a": [4.508200, 0.084888, 0.195520, 1.157000, 8.251700, 0.023700],
            "b": [2.070400E-08, 1.444900E-08, 1.597800E-08, 4.768600E-06, 1.112700E-05],
        },
        "ne": {  # 異常光線 (extraordinary ray)
            "a": [4.561500, 0.084880, 0.192700, 5.583200, 8.306700, 0.021696],
            "b": [4.782000E-07, 3.091300E-08, 2.732600E-08, 1.483700E-05, 1.364700E-07],
        },
    }["ne"]

    def _n_eff(wavelength: Wavelength, t: T) -> jnp.ndarray:
        f = (t - 24.5) * (t + 24.5 + 2 * 273.16)
        # セルマイヤーの分散式による計算
        lambda_sq = wavelength ** 2
        a = params["a"]
        b = params["b"]
        n_sq = (
            a[0] + b[0] * f +
            (a[1] + b[1] * f) / (lambda_sq - (a[2] + b[2] * f) ** 2) +
            (a[3] + b[3] * f) / (lambda_sq - (a[4] + b[4] * f) ** 2) -
            a[5] * lambda_sq
        )

        # 実行屈折率Nの計算
        N = jnp.sqrt(n_sq)
        return N

    N_omega = _n_eff(wavelength, t)
    N_2omega = _n_eff(wavelength / 2, t)

    beta_omega = 2 * jnp.pi * N_omega / wavelength
    beta_2omega = 2 * jnp.pi * N_2omega / (wavelength / 2)

    return lambda z: (beta_2omega - 2 * beta_omega) * z
