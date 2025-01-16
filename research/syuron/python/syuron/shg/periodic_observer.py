import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


class PeriodicObserver:
    def __init__(self, n_layers, width):
        """
        ある座標系において、すべての層が微小であるような任意の多層構造に見えている実体を考える
        同じ実体が、すべての幅が最初の幅と等しい一定幅の多層構造に見えている座標系を使った、計算を行うためのクラス
        """
        # n_layersは座標変換しても変わらない
        self.n_layers = n_layers
        # 周期構造に見える座標系での値
        self.domain_size = width * n_layers
        self.width = width
        self.dz = width

    def observe_from_transformed(self, transform_func: Callable[[np.ndarray], np.ndarray]):
        """
        既に一定幅に見えている多層構造において、座標変換を適用した後にどのような幅に見えるかを計算する
        """
        axis_transformed = np.linspace(0, self.domain_size, self.n_layers)

        # 座標変換を適用
        axis_periodic = transform_func(axis_transformed)

        # 幅配列を計算したいので後方差分をとる
        dz_dz_prime = np.diff(axis_periodic) / np.diff(axis_transformed)
        widths = self.width / dz_dz_prime

        return widths

    def infer_transformation(self, widths):
        """
        与えられた任意の幅の配列が、一定幅に見えている座標系からの、座標変換を逆算する
        """
        # 座標変換の微分 dz/dz' を計算
        dz_dz_prime = self.width / widths

        # 後方差分の場合は初期値が欠けているため、0 を先頭に追加
        dz_dz_prime = np.insert(dz_dz_prime, 0, 0)  # 初期値を 0 として追加

        # 座標変換を数値積分で復元
        discrete_transform = np.cumsum(dz_dz_prime * self.dz)

        return discrete_transform


def compare_transformation(periodic_observer: PeriodicObserver, result_array, target_func, title=""):
    """
    計算結果と目標関数を比較し、グラフを描画する。

    Parameters:
        result_array (np.array): 計算結果
        target_func (function): 目標関数
        title (str): グラフのタイトル
    """
    axis = np.linspace(0, periodic_observer.domain_size,
                       periodic_observer.n_layers)
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
