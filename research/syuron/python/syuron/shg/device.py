from typing import Protocol, Callable, Iterator, Tuple, List
import math


class Device(Protocol):
    """SHGデバイス

    SHGデバイスではphase_mismatchとkappaの二つが定義されています

    phase_mismatch: 位相不整合
    kappa: 非線形結合係数
    z_mesh: z軸のメッシュとそれぞれのステップ幅
    """

    def phase_mismatch(self, wavelength: float, T: float) -> Callable[[float], float]:
        ...

    def kappa(self, z: float) -> complex:
        ...

    def z_mesh(self) -> Iterator[Tuple[float, float]]:
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

    def __init__(self, widths: List[float], kappa_magnitude: float = 1.0):
        """
        周期分極構造の幅リストからPPMgOSLTデバイスを初期化する

        Args:
            widths: 分極ドメインの幅リスト (m)
            kappa_magnitude: 非線形結合係数の大きさ
        """
        import math

        self.widths = widths
        self.L = sum(widths)

        # メッシュの設定
        self.steps = 100000
        self.h = self.L / self.steps

        # z値とkappa値を事前計算
        self._z_values = [i * self.h for i in range(self.steps)]
        self._kappa_values = [complex(0, 0)] * self.steps

        # 各ドメインのkappa値を設定
        current_pos = 0
        domain_idx = 0

        for width in widths:
            next_pos = current_pos + width

            # このドメインに含まれるz点のインデックス範囲
            start_idx = math.floor(current_pos / self.h)
            end_idx = math.ceil(next_pos / self.h)

            # このドメインのkappa値を設定
            sign = 1 if domain_idx % 2 == 0 else -1
            for idx in range(start_idx, min(end_idx, self.steps)):
                self._kappa_values[idx] = complex(sign * kappa_magnitude, 0)

            current_pos = next_pos
            domain_idx += 1

    def _n_eff(self, wavelength: float, T: float) -> float:
        f = (T - 24.5) * (T + 24.5 + 2 * 273.16)
        # セルマイヤーの分散式による計算
        lambda_sq = wavelength ** 2
        a = self.params["a"]
        b = self.params["b"]
        n_sq = (
            a[0] + b[0] * f +
            (a[1] + b[1] * f) / (lambda_sq - (a[2] + b[2] * f) ** 2) +
            (a[3] + b[3] * f) / (lambda_sq - (a[4] + b[4] * f) ** 2) -
            a[5] * lambda_sq
        )

        # 実行屈折率Nの計算
        N = math.sqrt(n_sq)
        return N

    def phase_mismatch(self, wavelength: float, T: float) -> Callable[[float], float]:
        N_omega = self._n_eff(wavelength, T)
        N_2omega = self._n_eff(wavelength / 2, T)

        beta_omega = 2 * math.pi * N_omega / wavelength
        beta_2omega = 2 * math.pi * N_2omega / (wavelength / 2)

        return lambda z: (beta_2omega - 2 * beta_omega) * z

    def kappa(self, z: float) -> complex:
        """
        指定されたz位置での非線形結合係数を返す
        """
        idx = int(z / self.h)
        return self._kappa_values[idx]

    def z_mesh(self) -> Iterator[Tuple[float, float]]:
        """
        シミュレーションのためのz軸メッシュを生成
        """
        for z in self._z_values:
            yield (z, self.h)
