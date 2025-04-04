# 前提条件

SHGデバイスでは以下の非線形結合モード方程式が成り立つ

---
$$
\frac {d}{dz} A^\omega(z) = -j\boldsymbol\kappa^*[A^\omega(z)]^*A^{2\omega}(z)\exp(-j(2\Delta)z) \tag{2.59a}
$$
$$
\frac {d}{dz} A^{2\omega}(z) = -j\boldsymbol\kappa[A^\omega(z)]^2\exp(+j(2\Delta)z) \tag{2.59b}
$$
---

ここで

---
$$
2\Delta = \beta^{2\omega} - 2\beta^\omega \tag{2.60}
$$
$$
\boldsymbol\kappa (z) = \frac {2\omega\epsilon_0}{4} \iint [\boldsymbol E^{2\omega}(x,y)^*] d(x,y,z) [\boldsymbol E^\omega(x,y)]^2 dxdy \tag{2.61}
$$
---

である。

縦型QPMデバイスでは $\kappa(z)$ はドメインごとに符号のみが反転することから

---
$$
\kappa(z) = \kappa_{mag}(-1)^{n(z)}
$$
* $\kappa_{mag}$: $\kappa$ の大きさ
---

と表せる。

ここで

---
$$
n(z) = \sum_{m=0}^{\infty} m \cdot [\theta(z - z_m) - \theta(z - z_{m+1})]
$$
* $\theta$: ヘヴィサイド関数
* $z_0, z_1, z_2, ..., z_m, ...$: ドメイン境界の位置を順に並べた数列
---

である。

# 命題: 任意の構造での変換効率の導出

## 補題: passiveな座標変換とactiveな座標変換は同値
