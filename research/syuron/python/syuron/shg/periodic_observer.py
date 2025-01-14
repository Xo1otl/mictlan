import numpy as np
import matplotlib.pyplot as plt


class PeriodicObserver:
    def __init__(self, resolution, domain_size, period):
        """
        層の反転構造を観測するための座標系を初期化する。

        Parameters:
            resolution (int): 離散点の数
            domain_size (float): 座標系の領域サイズ
            period (float): 層の基底周期
        """
        # 座標系が変形してもこれらの値は変わらない
        self.resolution = resolution
        self.domain_size = domain_size
        self.period = period
        self.dz = domain_size / resolution

    def observe_from_transformed(self, transform_func):
        """
        与えられた座標系から見た層の幅を計算する。

        Parameters:
            transform_func (function): 座標変換関数 z = f(z')

        Returns:
            distribution (np.array): 非周期的な層の幅
        """
        z_prime = np.linspace(0, self.domain_size, self.resolution)

        # 座標変換を適用
        z = transform_func(z_prime)

        # 非周期的な層の幅を計算
        dz_dz_prime = np.gradient(z, z_prime)  # dz/dz' を中央差分で計算
        distribution = self.period / dz_dz_prime

        return distribution

    # FIXME: ここで与えられるのは幅配列なので修正が必要
    def infer_transform(self, distribution):
        """
        与えられた非周期的な幅の配列が周期的に見えている座標系からの座標変換を逆算する。

        Parameters:
            distribution (np.array): 非周期的な層の幅

        Returns:
            discrete_transform (np.array): 逆算された座標変換
        """
        # 座標変換の微分 dz/dz' を計算
        dz_dz_prime = self.period / distribution

        # 座標変換を数値積分で復元
        discrete_transform = np.cumsum(dz_dz_prime) * self.dz

        return discrete_transform

    def compare(self, result_array, target_func, title=""):
        """
        計算結果と目標関数を比較し、グラフを描画する。

        Parameters:
            result_array (np.array): 計算結果
            target_func (function): 目標関数
            title (str): グラフのタイトル
        """
        axis = np.linspace(0, self.domain_size, self.resolution)
        plt.figure(figsize=(8, 6))
        plt.plot(axis, result_array, label="Computed Result",
                 linestyle="-", linewidth=2)
        plt.plot(axis, target_func(axis), label="Target Function",
                 linestyle="--", linewidth=2)
        plt.xlabel("z'")
        plt.ylabel("Value")
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()
