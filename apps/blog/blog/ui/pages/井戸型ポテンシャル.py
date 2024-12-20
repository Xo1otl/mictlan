import streamlit as st
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'Noto Sans CJK JP'

st.set_page_config(layout="wide")

r"""
# 一次元の井戸型ポテンシャルの簡単な理解・前半

「0からLの間に量子が存在します。」

いま0からLの範囲に存在するっていったね！？

つまり0より前、Lより後ろには存在しないわけです

量子の存在確率は連続していなければならないという法則があります(唐突)

つまり$x=-0.000001$の位置では存在確率0％なのに、$x=0$で急に30%になるとかはありえません

量子の存在確率の確率密度関数は波の形をしているという物理法則があります(唐突)

確率密度関数は物理とは関係無い数学の概念です、とりあえずいろんな位置での存在確率を表す値をプロットすると、波のような形になるという物理法則があります(異論は認めない)

波というのは紐を振動させた時とか、海でみかける形のことで、以下の関数はそのような形をしています

$$
\frac{\partial^2 \psi}{\partial t^2} = a \frac{\partial^2 \psi}{\partial x^2}
$$

実はこの式の形が波っぽいというより、この式が波の定義です(波動方程式と呼ばれる)

とにかく量子の存在確率を表す確率密度関数は波動方程式であるという物理法則があります

以上を踏まえると、物理法則と前提条件を満たす波の形は以下の図のような感じ(どれでもいい)👇
"""


def plot_waves():
    L = 10  # x軸の範囲 (0, L)
    x = np.linspace(-2, L + 2, 500)

    # 波の式を計算 (パラメータはハードコード, 境界条件を満たすように調整)
    y1 = 2 * np.sin(1 * np.pi * x / L)  # 基本振動
    y1[x < 0] = 0
    y1[x > L] = 0

    y2 = 1.5 * np.sin(2 * np.pi * x / L)  # 2倍振動
    y2[x < 0] = 0
    y2[x > L] = 0

    y3 = 1 * np.sin(3 * np.pi * x / L)  # 3倍振動
    y3[x < 0] = 0
    y3[x > L] = 0

    # プロット
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x, y1, linestyle="--", label="Wave 1")
    plt.plot(x, y2, linestyle="--", label="Wave 2")
    plt.plot(x, y3, linestyle="--", label="Wave 3")
    plt.xlabel("位置x")
    plt.ylabel("確率密度y")
    plt.xlim(-2, L + 2)
    plt.ylim(-3, 3)  # 振幅の最大値に合わせて調整

    # 軸の数字を非表示にする
    plt.xticks([])
    plt.yticks([])

    st.pyplot(fig)


# 関数の実行
plot_waves()

r"""
## まとめ

* 量子の存在確率は連続して変化する
* 量子の確率密度関数は波の形をしている
"""

r"""
# 一次元の井戸型ポテンシャルの簡単な理解・後半

上で書いた **波の形**(確率密度関数の形) とその **量子の運動量やエネルギー** の間で、以下の式が成り立つという物理法則があります！(異論は認めない)

$$
\begin{aligned}
p &= \hbar k \\
E &= \frac{p^2}{2m} = \frac{\hbar^2 k^2}{2m} \\
\end{aligned}
$$
ここで、
$$
\begin{aligned}
&p：\text{粒子の運動量} \\
&\hbar：1.054571817 \times 10^{-34}\text{J} \cdot \text{s}\text{ぐらいの定数} \\
&k：\text{波数} \ (k = \frac{2\pi}{\lambda}, \ \lambda\text{は波長}) \\
&E：\text{粒子のエネルギー} \\
&m：\text{粒子の質量}
\end{aligned}
$$

この式から、上の図の青色の確率密度関数の量子の運動量とか、確率密度関数が赤のときの量子のエネルギーとかが求まります。

細かい計算は学校でならってください！簡単な理解は以上です。

量子力学やってて何が前提で何が導かれているのかさっぱりわからくなったことがあるので、同じような人の手助けになれば幸いです...orz
"""
