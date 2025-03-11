from typing import Protocol, Callable
import jax.numpy as jnp
import matplotlib.pyplot as plt


class Device(Protocol):
    """SHGデバイス

    phase_mismatchとkappaとz_meshが定義されているオブジェクトをSHGデバイスとして扱う。

    phase_mismatch: wavelengthとTに対応する、zに依存する位相不整合関数
    kappa: zに依存する非線形結合係数
    z_mesh: z軸のメッシュとそれぞれのステップ幅
    """

    def phase_mismatch(self, wavelength: jnp.ndarray, T: jnp.ndarray) -> Callable[[float], jnp.ndarray]:
        ...

    def kappa(self, z: float) -> jnp.ndarray:
        """
        TODO: zをndarrayで受け取れるようにする
        """
        ...

    def z_mesh(self) -> jnp.ndarray:
        """
        (z,dz)のペアのndarray
        """
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

    def __init__(self, widths: jnp.ndarray, kappa_magnitude: float = 1.0):
        """
        周期分極構造の幅リストからPPMgOSLTデバイスを初期化する

        Args:
            widths: 分極ドメインの幅リスト (m)
            kappa_magnitude: 非線形結合係数の大きさ
        """

        self.widths = widths
        self.kappa_magnitude = kappa_magnitude
        self.L = sum(widths)

        # メッシュの設定
        self.steps = 100000
        self.h = self.L / self.steps
        print(self.h)

        # z値とkappa値を事前計算
        self._z_indices = jnp.arange(self.steps) * self.h
        self._kappa_values = self._generate_kappa_values()

    def _generate_kappa_values(self):
        # 累積幅を計算
        cum_widths = jnp.cumsum(self.widths)
        domain_boundaries = jnp.concatenate([jnp.array([0.0]), cum_widths])

        # 各ステップが属するドメインのインデックスを計算
        domain_indices = jnp.searchsorted(
            domain_boundaries, self._z_indices, side='right') - 1

        # 符号を決定（偶数インデックスは+1、奇数は-1）
        signs = jnp.where(domain_indices % 2 == 0, 1.0, -1.0)
        return signs * self.kappa_magnitude

    def _n_eff(self, wavelength: jnp.ndarray, T: jnp.ndarray) -> jnp.ndarray:
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
        N = jnp.sqrt(n_sq)
        return N

    def phase_mismatch(self, wavelength: jnp.ndarray, T: jnp.ndarray) -> Callable[[float], jnp.ndarray]:
        N_omega = self._n_eff(wavelength, T)
        N_2omega = self._n_eff(wavelength / 2, T)

        beta_omega = 2 * jnp.pi * N_omega / wavelength
        beta_2omega = 2 * jnp.pi * N_2omega / (wavelength / 2)

        return lambda z: (beta_2omega - 2 * beta_omega) * z

    def kappa(self, z: float) -> jnp.ndarray:
        idx = jnp.minimum(jnp.floor(z / self.h).astype(int), self.steps-1)
        return self._kappa_values[idx]

    def z_mesh(self) -> jnp.ndarray:
        z = jnp.linspace(0, self.L, self.steps)
        dz = jnp.full(self.steps, self.h)
        return jnp.stack([z, dz], axis=1)
