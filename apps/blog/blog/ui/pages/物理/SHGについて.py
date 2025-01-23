import streamlit as st

r"""
# SHGについて

修論でやってた縦型疑似位相整合で出てきたODEについて考察します

まとめ直すことにより理論があってるかの確認も兼ねる

## 連立ODE

Nonlinear Coupled Mode Equations (NCME)
$$
\frac{d}{dz} A(z) = -j \kappa ^ * A^*(z) B(z) exp(j 2 \Delta_1 (z)) \\
\frac{d}{dz} B(z) = -j \kappa [A(z)]^2 exp(+j 2 \Delta_1 (z)) \\
$$
ただし
$$
2\Delta_q(z) = \beta_{2\omega} z - 2\beta_{\omega} z - Kz \\
K = \frac{2\pi}{\Lambda}
$$
ここで$Kz$は
$$
Kz = \int K(z) dz
$$
から来る式で
$$
x=x', y=y', z=z'+(r/2)z'^2
$$
という座標変換を行うと

微分では、微小幅dzを設定し、その幅で分割を行って計算を行う

同じ微小幅が、z座標系とz'座標系でそれぞれdz, dz'という見え方だったとする

同様にして$\Lambdaと\Lambda'$も、同じ微小幅を二つの座標系から見たときの値である

よって、$dz : dz' = \Lambda : \Lambda'$

## プログラムの設計

### PeriodicObserver

任意の幅配列を持つ多層構造に見えている実体が、幅配列の最初の要素の幅だけの一定幅構造に見えている座標系に住んでいるObserver

### NCME

連立ODE

### Grating
"""
