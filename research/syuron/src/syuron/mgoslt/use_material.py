import jax.numpy as jnp
from syuron import shg


def use_material(wavelength: shg.Wavelength, t: shg.T) -> shg.PhaseMismatchFn:
    """
    MgO:SLT材料における、指定された波長と温度での位相不整合関数を計算します。

    この関数は、MgO:SLTの異常光線(extraordinary ray)に対する
    温度依存のセルマイヤー方程式を用いて、基本波(ω)と
    第二高調波(2ω)の屈折率を計算し、それらから単位長さあたりの
    位相不整合 Δβ = β(2ω) - 2β(ω) を求めます。

    Args:
        wavelength: 基本波の波長 (µm)。
        t: 材料の温度 (摂氏)。

    Returns:
        位相不整合関数 (shg.PhaseMismatchFn)。
        この関数は距離 `z` (µm) を引数に取り、
        その地点での累積位相不整合量 `Δβ * z` (ラジアン) を返します。
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

    def _n_eff(wavelength: shg.Wavelength, t: shg.T) -> jnp.ndarray:
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
    phase_mismatch = beta_2omega - 2 * beta_omega

    return lambda z: phase_mismatch * z
