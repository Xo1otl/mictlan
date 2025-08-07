### **結合波方程式**

カスケード二次非線形過程（SHG+SFG）による第三高調波発生（THG）は、以下の結合波方程式系で記述される。

$$\frac{d\boldsymbol{A}}{dz} = \boldsymbol{f}(\boldsymbol{A}, z; \kappa(z)), \quad \boldsymbol{A}(z) = [A_1(z), A_2(z), A_3(z)]^T$$

$$
\begin{align}
\frac{d A_1}{dz} &= i \kappa(z) \left[ A_2 A_1^* e^{i\Delta k_1 z} + A_3 A_2^* e^{i\Delta k_2 z} \right] \tag{1} \\
\frac{d A_2}{dz} &= i \, 2\kappa(z) \left[ \frac{1}{2} A_1^2 e^{-i\Delta k_1 z} + A_3 A_1^* e^{i\Delta k_2 z} \right] \tag {2} \\
\frac{d A_3}{dz} &= i \, 3\kappa(z) \left[ A_1 A_2 e^{-i\Delta k_2 z} \right] \tag{3}
\end{align}
$$

ここで、$\boldsymbol{A}(z)$ は各波の複素振幅ベクトル、$\kappa(z)$ は結合係数、$\Delta k_j$ は位相不整合量を表す。

本問題では、分極反転素子は複数の**domain**から構成されるものとする。各ドメインの境界を**domain wall**と呼ぶ。結合係数 $\kappa(z)$ は、素子全体でその絶対値 $|\kappa(z)|$ が一定値 $\kappa_0$ をとり、ドメインごとに符号のみが変化する区分一定関数であるとする。

$k$番目のドメイン $[z_k, z_{k+1}]$ における伝播は、始点での振幅ベクトル $\boldsymbol{A}(z_k)$ を終点での振幅ベクトル $\boldsymbol{A}(z_{k+1})$ に写す非線形な伝播演算子（Propagator）$\mathcal{P}_k$ を用いて、$\boldsymbol{A}(z_{k+1}) = \mathcal{P}_k \left[ \boldsymbol{A}(z_k) \right]$ と定義する。
素子全体（$z=0$ から $z=L$）の伝播は、各区間の伝播演算子を順次適用することで与えられ、その形式解は以下のように記述される。

$$\boldsymbol{A}(L) = \left( \prod_{k=N-1}^{0} \mathcal{P}_k \right) \boldsymbol{A}(0)$$

### **目標**

与えられた結合波方程式系を、剛直な（Stiff）線形部分と、非剛直な非線形部分に分離した半線形（Semilinear）形式
$$\frac{d\boldsymbol{B}}{dz} = \boldsymbol{L}\boldsymbol{B} + \boldsymbol{N}(\boldsymbol{B})$$
へ変換し、積分形式（定数変化公式）を導出する。

### **ステップ1：相互作用表示（Interaction Picture）への変換**

剛直性の原因は、位相不整合項 $e^{i\Delta k_j z}$ に含まれる高速な振動である。この振動部分を系のダイナミクスから分離するために、適切な変数変換を行う。

#### **変数変換の導入**

高速に振動する指数関数部分を吸収するため、新しい変数ベクトル $\boldsymbol{B}(z) = [B_1(z), B_2(z), B_3(z)]^T$ を以下のように定義する。

$$
\begin{align}
A_1(z) &= B_1(z) \tag{4} \\
A_2(z) &= B_2(z) e^{-i\Delta k_1 z} \tag{5} \\
A_3(z) &= B_3(z) e^{-i(\Delta k_1 + \Delta k_2) z} \tag{6}
\end{align}
$$

この変換により、$\boldsymbol{B}(z)$ は $\boldsymbol{A}(z)$ から高速な振動を取り除いた、より緩やかに変化する量となることが期待される。

#### **微分方程式の変換（詳細な計算）**

積の微分法則 $\frac{d}{dz}(fg) = \frac{df}{dz}g + f\frac{dg}{dz}$ を用いて、各 $B_k$ の微分方程式を導出する。

**1. $B_1$ の方程式**

式(4)を $z$ で微分する。
$$\frac{dA_1}{dz} = \frac{dB_1}{dz}$$
これに式(1)を代入する。
$$\frac{dB_1}{dz} = i \kappa_k \left[ A_2 A_1^* e^{i\Delta k_1 z} + A_3 A_2^* e^{i\Delta k_2 z} \right]$$
右辺の $A_k$ を $B_k$ を用いて書き換える（式(4)〜(6)を代入）。
$$
\begin{aligned}
\frac{dB_1}{dz} &= i \kappa_k \left[ (B_2 e^{-i\Delta k_1 z}) (B_1^*) e^{i\Delta k_1 z} + (B_3 e^{-i(\Delta k_1 + \Delta k_2) z}) (B_2^* e^{i\Delta k_1 z}) e^{i\Delta k_2 z} \right] \\
&= i \kappa_k \left[ B_2 B_1^* (e^{-i\Delta k_1 z} e^{i\Delta k_1 z}) + B_3 B_2^* (e^{-i\Delta k_1 z} e^{-i\Delta k_2 z} e^{i\Delta k_1 z} e^{i\Delta k_2 z}) \right] \\
&= i \kappa_k \left[ B_1^* B_2 + B_2^* B_3 \right] \tag{7}
\end{aligned}
$$
この方程式には、振動項 $e^{i\Delta k z}$ が現れなくなった。

**2. $B_2$ の方程式**

式(5)を $z$ で微分するために、まず $B_2(z) = A_2(z) e^{i\Delta k_1 z}$ と変形する。
$$
\begin{aligned}
\frac{dB_2}{dz} &= \frac{dA_2}{dz} e^{i\Delta k_1 z} + A_2 \frac{d}{dz}(e^{i\Delta k_1 z}) \\
&= \frac{dA_2}{dz} e^{i\Delta k_1 z} + i\Delta k_1 A_2 e^{i\Delta k_1 z} \\
&= \frac{dA_2}{dz} e^{i\Delta k_1 z} + i\Delta k_1 B_2
\end{aligned}
$$
ここで、元の $\frac{dA_2}{dz}$（式(2)）を代入する。
$$
\begin{aligned}
\frac{dB_2}{dz} &= \left( i \, 2\kappa_k \left[ \frac{1}{2} A_1^2 e^{-i\Delta k_1 z} + A_3 A_1^* e^{i\Delta k_2 z} \right] \right) e^{i\Delta k_1 z} + i\Delta k_1 B_2 \\
&= i \, 2\kappa_k \left[ \frac{1}{2} A_1^2 (e^{-i\Delta k_1 z} e^{i\Delta k_1 z}) + A_3 A_1^* (e^{i\Delta k_2 z} e^{i\Delta k_1 z}) \right] + i\Delta k_1 B_2 \\
&= i \kappa_k A_1^2 + i \, 2\kappa_k A_3 A_1^* e^{i(\Delta k_1 + \Delta k_2) z} + i\Delta k_1 B_2
\end{aligned}
$$
右辺の $A_k$ を $B_k$ で書き換える。
$$
\begin{aligned}
\frac{dB_2}{dz} &= i\Delta k_1 B_2 + i \kappa_k B_1^2 + i \, 2\kappa_k (B_3 e^{-i(\Delta k_1 + \Delta k_2) z}) B_1^* e^{i(\Delta k_1 + \Delta k_2) z} \\
&= i\Delta k_1 B_2 + i \kappa_k \left[ B_1^2 + 2 B_1^* B_3 \right] \tag{8}
\end{aligned}
$$

**3. $B_3$ の方程式**

式(6)を $z$ で微分するために、まず $B_3(z) = A_3(z) e^{i(\Delta k_1 + \Delta k_2) z}$ と変形する。
$$
\begin{aligned}
\frac{dB_3}{dz} &= \frac{dA_3}{dz} e^{i(\Delta k_1 + \Delta k_2) z} + A_3 \frac{d}{dz}(e^{i(\Delta k_1 + \Delta k_2) z}) \\
&= \frac{dA_3}{dz} e^{i(\Delta k_1 + \Delta k_2) z} + i(\Delta k_1 + \Delta k_2) A_3 e^{i(\Delta k_1 + \Delta k_2) z} \\
&= \frac{dA_3}{dz} e^{i(\Delta k_1 + \Delta k_2) z} + i(\Delta k_1 + \Delta k_2) B_3
\end{aligned}
$$
ここで、元の $\frac{dA_3}{dz}$（式(3)）を代入する。
$$
\begin{aligned}
\frac{dB_3}{dz} &= \left( i \, 3\kappa_k A_1 A_2 e^{-i\Delta k_2 z} \right) e^{i(\Delta k_1 + \Delta k_2) z} + i(\Delta k_1 + \Delta k_2) B_3 \\
&= i \, 3\kappa_k A_1 A_2 e^{i\Delta k_1 z} + i(\Delta k_1 + \Delta k_2) B_3
\end{aligned}
$$
右辺の $A_k$ を $B_k$ で書き換える。
$$
\begin{align}
\frac{dB_3}{dz} &= i(\Delta k_1 + \Delta k_2) B_3 + i \, 3\kappa_k (B_1) (B_2 e^{-i\Delta k_1 z}) e^{i\Delta k_1 z} \\
&= i(\Delta k_1 + \Delta k_2) B_3 + i \, 3\kappa_k B_1 B_2 \tag{9}
\end{align}
$$

### **ステップ2：半線形自励系への整理**

導出した3つの方程式(7), (8), (9)をベクトルと行列を用いてまとめる。

#### **変換後の方程式系**

$$
\begin{align}
\frac{dB_1}{dz} &= i \kappa_k (B_1^* B_2 + B_2^* B_3) \\
\frac{dB_2}{dz} &= i\Delta k_1 B_2 + i \kappa_k (B_1^2 + 2 B_1^* B_3) \\
\frac{dB_3}{dz} &= i(\Delta k_1 + \Delta k_2) B_3 + i \, 3\kappa_k B_1 B_2
\end{align}
$$

これを $\frac{d\boldsymbol{B}}{dz} = \boldsymbol{L}\boldsymbol{B} + \boldsymbol{N}(\boldsymbol{B})$ の形に分離する。

* **線形項 $\boldsymbol{L}\boldsymbol{B}$**: 右辺のうち、$B_k$ に線形な項を集める。
* **非線形項 $\boldsymbol{N}(\boldsymbol{B})$**: 右辺の残りの項（$B_k$ の二次以上の項）を集める。

#### **ベクトル・行列形式での表現**

$$
\frac{d}{dz}
\begin{pmatrix} B_1 \\ B_2 \\ B_3 \end{pmatrix}
=
\underbrace{
i \begin{pmatrix}
0 & 0 & 0 \\
0 & \Delta k_1 & 0 \\
0 & 0 & \Delta k_1 + \Delta k_2
\end{pmatrix}
}_{\boldsymbol{L}}
\begin{pmatrix} B_1 \\ B_2 \\ B_3 \end{pmatrix}
+
\underbrace{
i \kappa_k \begin{pmatrix}
B_1^* B_2 + B_2^* B_3 \\
B_1^2 + 2 B_1^* B_3 \\
3 B_1 B_2
\end{pmatrix}
}_{\boldsymbol{N}(\boldsymbol{B})}
$$

これにより、方程式系は目標としていた半線形形式に変換された。$\boldsymbol{L}$は対角行列であり、剛直性の原因となる線形的な位相回転を記述する。$\boldsymbol{N}(\boldsymbol{B})$は波の間の非線形なエネルギー交換を記述する。

### **ステップ3：定数変化法による厳密解の表現**

得られた半線形方程式
$$\frac{d\boldsymbol{B}}{dz} = \boldsymbol{L}\boldsymbol{B} + \boldsymbol{N}(\boldsymbol{B})$$
を、区間 $[z_n, z_n+h]$ で積分することを考える。この1階線形非斉次方程式は、定数変化法を用いて積分できる。

まず、式を移項する。
$$\frac{d\boldsymbol{B}}{dz} - \boldsymbol{L}\boldsymbol{B} = \boldsymbol{N}(\boldsymbol{B})$$
この方程式の積分因子は $e^{-\boldsymbol{L}z}$ である。両辺に左から積分因子を掛ける。
$$e^{-\boldsymbol{L}z} \left( \frac{d\boldsymbol{B}}{dz} - \boldsymbol{L}\boldsymbol{B} \right) = e^{-\boldsymbol{L}z} \boldsymbol{N}(\boldsymbol{B})$$
行列指数関数の微分の性質 $\frac{d}{dz} e^{-\boldsymbol{L}z} = -\boldsymbol{L} e^{-\boldsymbol{L}z}$ と積の微分法則から、左辺は以下のようにまとめられる。
$$\frac{d}{dz} \left( e^{-\boldsymbol{L}z} \boldsymbol{B}(z) \right) = e^{-\boldsymbol{L}z} \boldsymbol{N}(\boldsymbol{B}(z))$$
この式を $z$ について区間 $[z_n, z_n+h]$ で定積分する。
$$\int_{z_n}^{z_n+h} \frac{d}{d\tau} \left( e^{-\boldsymbol{L}\tau} \boldsymbol{B}(\tau) \right) d\tau = \int_{z_n}^{z_n+h} e^{-\boldsymbol{L}\tau} \boldsymbol{N}(\boldsymbol{B}(\tau)) d\tau$$
左辺を計算すると、
$$\left[ e^{-\boldsymbol{L}\tau} \boldsymbol{B}(\tau) \right]_{z_n}^{z_n+h} = e^{-\boldsymbol{L}(z_n+h)} \boldsymbol{B}(z_n+h) - e^{-\boldsymbol{L}z_n} \boldsymbol{B}(z_n)$$
よって、
$$e^{-\boldsymbol{L}(z_n+h)} \boldsymbol{B}(z_n+h) - e^{-\boldsymbol{L}z_n} \boldsymbol{B}(z_n) = \int_{z_n}^{z_n+h} e^{-\boldsymbol{L}\tau} \boldsymbol{N}(\boldsymbol{B}(\tau)) d\tau$$
$\boldsymbol{B}(z_n+h)$ について解くために、両辺に左から $e^{\boldsymbol{L}(z_n+h)}$ を掛ける。
$$\boldsymbol{B}(z_n+h) - e^{\boldsymbol{L}h} \boldsymbol{B}(z_n) = e^{\boldsymbol{L}(z_n+h)} \int_{z_n}^{z_n+h} e^{-\boldsymbol{L}\tau} \boldsymbol{N}(\boldsymbol{B}(\tau)) d\tau$$
積分変数を $\tau' = \tau - z_n$ と置換すると、$d\tau' = d\tau$ であり、積分範囲は $[0, h]$ となる。
$$
\begin{aligned}
\boldsymbol{B}(z_n+h) &= e^{\boldsymbol{L}h} \boldsymbol{B}(z_n) + e^{\boldsymbol{L}(z_n+h)} \int_0^h e^{-\boldsymbol{L}(z_n+\tau')} \boldsymbol{N}(\boldsymbol{B}(z_n+\tau')) d\tau' \\
%
&\quad \downarrow \text{定数項 } e^{\boldsymbol{L}(z_n+h)} \text{ を積分の中へ移動} \\
&= e^{\boldsymbol{L}h} \boldsymbol{B}(z_n) + \int_0^h e^{\boldsymbol{L}(z_n+h)} e^{-\boldsymbol{L}(z_n+\tau')} \boldsymbol{N}(\boldsymbol{B}(z_n+\tau')) d\tau' \\
%
&\quad \downarrow \text{指数法則を適用} \\
&= e^{\boldsymbol{L}h} \boldsymbol{B}(z_n) + \int_0^h e^{\boldsymbol{L}(z_n+h - (z_n+\tau'))} \boldsymbol{N}(\boldsymbol{B}(z_n+\tau')) d\tau' \\
%
&= e^{\boldsymbol{L}h} \boldsymbol{B}(z_n) + \int_0^h e^{\boldsymbol{L}(h-\tau')} \boldsymbol{N}(\boldsymbol{B}(z_n+\tau')) d\tau'
\end{aligned}
$$

### **結論**

Exponential Time Differencing (ETD) 法は、この厳密な積分形式から出発します。右辺の積分は、被積分関数の中に未知の解 $\boldsymbol{B}(z_n+\tau')$ が含まれているため直接計算できません。ETD法は、この積分内の非線形項 $\boldsymbol{N}(\boldsymbol{B}(z_n+\tau'))$ を、既知の点 $z_n$ における値 $\boldsymbol{N}(\boldsymbol{B}(z_n))$ などを用いて多項式で近似し、積分を解析的または数値的に評価することで、$\boldsymbol{B}(z_n+h)$ の近似値を計算する手法の総称です。この定式化により、剛直な線形項 $e^{\boldsymbol{L}h}$ は厳密に計算されるため、数値計算は非常に安定します。
