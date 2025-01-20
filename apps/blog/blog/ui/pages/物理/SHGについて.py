import streamlit as st

r"""
# SHGについて

修論でやってた縦型疑似位相整合で出てきたODEについて考察します、座標変換による式変形がめっちゃエレガント

まとめ直すことにより理論があってるかの確認も兼ねる

## 連立ODE

Nonlinear Coupled Mode Equations (NCME)
$$
\frac{d}{dz} A(z) = -j \kappa ^ * A^*(z) B(z) exp(j 2 \Delta_1 (z)) \\
\frac{d}{dz} B(z) = -j \kappa [A(z)]^2 exp(+j 2 \Delta_1 (z)) \\
$$
ただし
$$
2\Delta_q(z) = \beta_{2\omega} z - 2\beta_{\omega} z - q \Phi(z) \\
$$

## プログラムの設計

### PeriodicObserver

任意の幅配列を持つ多層構造に見えている実体が、幅配列の最初の要素の幅だけの一定幅構造に見えている座標系に住んでいるObserver

### NCME

連立ODE

### Grating
"""
