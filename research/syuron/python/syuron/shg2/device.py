import math
from typing import Protocol, Callable


class Device(Protocol):
    def phase_mismatch(self, wavelength: float) -> Callable[[float], float]:
        ...

    def kappa(self, z: float) -> complex:
        ...


class PPMgOSLT(Device):
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

    def __init__(self, z_vals: list[float], T=24.5):
        self.T = T
        self.f = (T - 24.5) * (T + 24.5 + 2 * 273.16)

    def _n_eff(self, wavelength: float) -> float:
        # セルマイヤーの分散式による計算
        lambda_sq = wavelength ** 2
        a = self.params["a"]
        b = self.params["b"]
        n_sq = (
            a[0] + b[0] * self.f +
            (a[1] + b[1] * self.f) / (lambda_sq - (a[2] + b[2] * self.f) ** 2) +
            (a[3] + b[3] * self.f) / (lambda_sq - (a[4] + b[4] * self.f) ** 2) -
            a[5] * lambda_sq
        )

        # 実行屈折率Nの計算
        N = math.sqrt(n_sq)
        return N

    def phase_mismatch(self, wavelength: float) -> Callable[[float], float]:
        N_omega = self._n_eff(wavelength)
        N_2omega = self._n_eff(wavelength / 2)

        beta_omega = 2 * math.pi * N_omega / wavelength
        beta_2omega = 2 * math.pi * N_2omega / (wavelength / 2)

        return lambda z: (beta_2omega - 2 * beta_omega) * z
