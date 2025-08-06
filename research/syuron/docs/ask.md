## 質問

カスケード二次非線形過程（SHG+SFG）による第三高調波発生（THG）は、以下の結合波方程式系で記述される。

$$\frac{d\boldsymbol{A}}{dz} = \boldsymbol{f}(\boldsymbol{A}, z; \kappa(z)), \quad \boldsymbol{A}(z) = [A_1(z), A_2(z), A_3(z)]^T$$

$$
\begin{aligned}
\frac{d A_1}{dz} &= i \kappa(z) \left[ A_2 A_1^* e^{i\Delta k_1 z} + A_3 A_2^* e^{i\Delta k_2 z} \right] \\
\frac{d A_2}{dz} &= i \, 2\kappa(z) \left[ \frac{1}{2} A_1^2 e^{-i\Delta k_1 z} + A_3 A_1^* e^{i\Delta k_2 z} \right] \\
\frac{d A_3}{dz} &= i \, 3\kappa(z) \left[ A_1 A_2 e^{-i\Delta k_2 z} \right]
\end{aligned}
$$

ここで、$\boldsymbol{A}(z)$ は各波の複素振幅ベクトル、$\kappa(z)$ は結合係数、$\Delta k_j$ は位相不整合量を表す。

本問題では、分極反転素子は複数の**domain**から構成されるものとする。各ドメインの境界を**domain wall**と呼ぶ。結合係数 $\kappa(z)$ は、素子全体でその絶対値 $|\kappa(z)|$ が一定値 $\kappa_0$ をとり、ドメインごとに符号のみが変化する区分一定関数であるとする。

$k$番目のドメイン $[z_k, z_{k+1}]$ における伝播は、始点での振幅ベクトル $\boldsymbol{A}(z_k)$ を終点での振幅ベクトル $\boldsymbol{A}(z_{k+1})$ に写す非線形な伝播演算子（Propagator）$\mathcal{P}_k$ を用いて、$\boldsymbol{A}(z_{k+1}) = \mathcal{P}_k \left[ \boldsymbol{A}(z_k) \right]$ と定義する。
素子全体（$z=0$ から $z=L$）の伝播は、各区間の伝播演算子を順次適用することで与えられ、その形式解は以下のように記述される。

$$\boldsymbol{A}(L) = \left( \prod_{k=N-1}^{0} \mathcal{P}_k \right) \boldsymbol{A}(0)$$

以下の問に答えよ。

### **問1**

単一ドメイン内における上記方程式系の数値解法、すなわち伝播演算子 $\mathcal{P}_k$ の具体的な計算手法について考察する。

(1) Exponential Integratorの枠組みを適用し、本方程式系を解くための具体的な数学的定式化を導出せよ。

(2) (1)で導出した手法と、標準的な陽的ルンゲ＝クッタ法を比較し、数値安定性および計算効率の観点から両者の精度における優劣を数学的に証明せよ。

### **問2**

素子の分極反転構造を最適化する逆問題を考える。構造はドメインウォールの位置の集合 $\{z_k\}$ によって定義されるものとし、目的関数を $J(\{z_k\})$ とする。

(1) 随伴法（Adjoint Method）を用いて、目的関数の各domain wall位置に対する勾配 $\partial J / \partial z_k$ が導出可能か、できるだけ厳密に示せ。

(2) 制御パラメータがドメインウォールの位置 $\{z_k\}$ である場合、目的関数 $J(\{z_k\})$ に対して勾配法に基づく最適化が適用可能であるか、その理論的根拠を論ぜよ。

(3) 本問題の最適化において、劣勾配（subgradient）の概念は必要か。標準的な勾配の枠組みで十分か。判断の根拠となる数学的条件や仮定を明確にして議論せよ。

## 回答

本問題は、非線形光学における結合波方程式系の数値解析と、デバイス構造最適化に関する高度な数学的解析を要求するものです。以下に、各問に対する詳細な回答を示します。

### 問1：単一ドメイン内における数値解法

与えられた結合波方程式系は、位相不整合項 $e^{i\Delta k_j z}$ により、特に $|\Delta k_j|$ が大きい場合に解が高速に振動する、数値的に剛直（Stiff）な性質を持ちます。

#### (1) Exponential Integratorの枠組みの適用と数学的定式化

Exponential Integrator（指数積分法）、特にExponential Time Differencing (ETD)法を適用するため、まず系を相互作用表示（Interaction Picture）へ変換し、剛直な線形成分と緩やかな非線形成分を分離します。

**相互作用表示への変換**

全位相不整合を $\Delta k_{THG} = \Delta k_1 + \Delta k_2$ と定義し、以下の変数変換を導入します。

$$
\begin{aligned}
A_1(z) &= B_1(z) \\
A_2(z) &= B_2(z) e^{-i\Delta k_1 z} \\
A_3(z) &= B_3(z) e^{-i\Delta k_{THG} z}
\end{aligned}
$$

単一ドメイン内（結合係数 $\kappa(z) = \kappa_k$ が一定）において、この変換により、ベクトル $\boldsymbol{B}(z) = [B_1, B_2, B_3]^T$ に対する方程式系は、以下の半線形（Semilinear）な自励系に帰着されます。

$$
\frac{d\boldsymbol{B}}{dz} = \boldsymbol{L}\boldsymbol{B} + \boldsymbol{N}(\boldsymbol{B})
$$

ここで、$\boldsymbol{L}$ は剛直性の原因となる線形演算子（対角行列）、$\boldsymbol{N}(\boldsymbol{B})$ は非線形項です。

$$
\boldsymbol{L} = i \begin{pmatrix}
0 & 0 & 0 \\
0 & \Delta k_1 & 0 \\
0 & 0 & \Delta k_{THG}
\end{pmatrix}, \quad
\boldsymbol{N}(\boldsymbol{B}) = i \kappa_k \begin{pmatrix}
B_2 B_1^* + B_3 B_2^* \\
B_1^2 + 2 B_3 B_1^* \\
3 B_1 B_2
\end{pmatrix}
$$

**ETD法の定式化**

区間 $[z_n, z_n+h]$ における厳密解は、定数変化公式（Variation-of-Constants formula）により与えられます。

$$
\boldsymbol{B}(z_n+h) = e^{\boldsymbol{L}h}\boldsymbol{B}(z_n) + \int_0^h e^{\boldsymbol{L}(h-\tau)} \boldsymbol{N}(\boldsymbol{B}(z_n+\tau)) d\tau
$$

ETD法は、この積分項を数値的に近似します。例えば、広く用いられる4次のETDルンゲ＝クッタ法（ETDRK4）では、非線形項 $\boldsymbol{N}(\boldsymbol{B})$ を複数点で評価し、それらを$\varphi$関数（$\varphi_0(z)=e^z, \varphi_{k+1}(z) = (\varphi_k(z) - 1/k!)/z$）の線形結合で重み付けすることで、積分を4次精度で近似します。$\boldsymbol{L}$が対角行列であるため、これらの行列関数の計算は容易です。これが伝播演算子 $\mathcal{P}_k$ の具体的な計算手法となります。

#### (2) Exponential Integratorと陽的ルンゲ＝クッタ法の比較と証明

標準的な陽的ルンゲ＝クッタ法（RK4）とETDRK4の優劣を、数値安定性と計算効率（精度）の観点から比較証明します。剛直性は、$\boldsymbol{L}$の最大固有値の絶対値 $\Lambda = \max(|\Delta k_1|, |\Delta k_{THG}|)$ が大きいことに起因します。

**数値安定性**

1.  **RK4:** RK4の絶対安定領域は有界であり、虚軸上では約 $[-2.83i, 2.83i]$ に制限されます。安定な計算には、ステップ幅 $h$ がCFL条件 $h\Lambda \lesssim 2.83$ を満たす必要があります。$\Lambda$ が大きい場合、$h$ は極めて小さく制限されます。
2.  **ETDRK4:** ETD法は線形項 $\boldsymbol{L}\boldsymbol{B}$ を厳密に積分します。$\boldsymbol{L}$の固有値は純虚数であるため、$e^{\boldsymbol{L}h}$ の絶対値は1となり、線形部分に対しては無条件に安定（A-stable）です。安定性は非線形項のダイナミクス（結合係数 $\kappa_0$ の大きさ）のみに依存します。

**精度と計算効率（誤差解析による証明）**

両手法は形式的に4次精度（局所誤差 $O(h^5)$）ですが、剛直な系では誤差定数の挙動が決定的に異なります。

1.  **RK4の誤差:** RK4を剛直な系に適用した場合、局所誤差 $E_{RK4}$ の定数は $\boldsymbol{L}$ の高次のべき乗に依存します。
    $$E_{RK4} = O(h^5 (\boldsymbol{L}^5 + \dots))$$
    誤差定数は $\Lambda$ の高次べき乗（例：$\Lambda^4$）に比例して増大します。精度を維持するためには、$h\Lambda$ を小さく保つ必要があり、安定性限界と同様の厳しい制限 $h \ll 1/\Lambda$ が課せられます（Order Reduction）。

2.  **ETDRK4の誤差:** HochbruckとOstermannらによる厳密な解析によれば、解が十分に滑らかであるという仮定の下で、ETDRK4の誤差定数は $\Lambda$ に依存しません。
    $$E_{ETDRK4} = O(h^5 \|\boldsymbol{N}^{(4)}\|)$$
    誤差は非線形相互作用の強さ（$\|\kappa_0\|$）に依存しますが、位相不整合 $\Lambda$ には依存しません。

**証明の結論:**
多くの物理系では、位相不整合は非線形結合よりもはるかに大きい（$\Lambda \gg \|\kappa_0\|$）、すなわち剛直な系です。この場合、RK4は $h \ll 1/\Lambda$ を要求しますが、ETDRK4は $h \ll 1/\|\kappa_0\|$ のみを要求します。したがって、ETDRK4はRK4よりもはるかに大きなステップ幅を採用でき、計算効率が著しく優れていることが数学的に証明されます。

### 問2：分極反転構造の最適化（逆問題）

ドメインウォールの位置 $\{z_k\}$ を最適化する逆問題を考えます。目的関数を $J(\{z_k\}) = G(\boldsymbol{A}(L))$ とします。

#### (1) 随伴法（Adjoint Method）による勾配 $\partial J / \partial z_k$ の導出可能性

随伴法を用いて勾配 $\partial J / \partial z_k$ を導出することは可能です。本問題は、制御パラメータが微分方程式の係数 $\kappa(z)$ が不連続に変化する位置 $z_k$ を決定するものであり、最適制御理論における**スイッチング時刻最適化（Switching Time Optimization; STO）**問題として定式化できます。

**随伴系の定義**

随伴変数ベクトル $\boldsymbol{\lambda}(z)$ を導入し、以下の随伴方程式系を定義します（$\dagger$はエルミート共役）。

$$
\frac{d\boldsymbol{\lambda}}{dz} = - \left( \frac{\partial \boldsymbol{f}}{\partial \boldsymbol{A}} \right)^\dagger \boldsymbol{\lambda}
$$

終端条件は $\boldsymbol{\lambda}(L) = \left( \frac{\partial G}{\partial \boldsymbol{A}(L)} \right)^*$ です。この方程式を $z=L$ から $z=0$ へ逆方向に解きます。

**勾配の導出（STO理論）**

STO理論によれば、スイッチング位置 $z_k$ に対する目的関数の勾配は、その位置におけるハミルトニアン $H(z)$ の跳躍（Jump）によって厳密に与えられます。ハミルトニアンを以下で定義します。

$$ H(z) = \text{Re}\left[ \boldsymbol{\lambda}^\dagger(z) \boldsymbol{f}(\boldsymbol{A}(z), z; \kappa(z)) \right] $$

勾配は以下で与えられます。

$$
\frac{\partial J}{\partial z_k} = H(z_k^-) - H(z_k^+)
$$

ここで、$z_k^-, z_k^+$ は $z_k$ への左側極限と右側極限です。状態変数 $\boldsymbol{A}(z)$ と随伴変数 $\boldsymbol{\lambda}(z)$ は $z_k$ において連続であるため、この跳躍は右辺の関数 $\boldsymbol{f}$ の不連続性のみに起因します。$z_k$ の前後で結合係数が $\kappa_{k-1}$ から $\kappa_k$ に変化するとします。

$$
\frac{\partial J}{\partial z_k} = \text{Re} \left[ \boldsymbol{\lambda}^\dagger(z_k) \cdot \left( \boldsymbol{f}(\dots; \kappa_{k-1}) - \boldsymbol{f}(\dots; \kappa_k) \right) \right]
$$

$\boldsymbol{f}$ は $\kappa$ に線形であるため、分極反転構造（$\kappa_{k-1} = -\kappa_k$）において、この差分はゼロにならず、有限の勾配値が得られます。この導出は、常微分方程式の解の存在と一意性が保証される限り、数学的に厳密です。

#### (2) 勾配法に基づく最適化の適用可能性とその理論的根拠

勾配法（最急降下法、準ニュートン法など）の適用は可能です。

**理論的根拠:**
勾配法の適用可能性は、目的関数 $J(\{z_k\})$ が制御パラメータ $\{z_k\}$ に関して連続的微分可能（$C^1$級）であることに依拠します。

その根拠は、常微分方程式の解のパラメータ依存性に関する定理にあります。方程式の右辺 $\boldsymbol{f}$ が状態変数 $\boldsymbol{A}$ に関して滑らか（リプシッツ連続）であれば（本問題では多項式であり満たされる）、その解 $\boldsymbol{A}(L)$ は、方程式を定義するパラメータに対して滑らかに依存します。

本問題において、係数 $\kappa(z)$ は $z$ に関して不連続ですが、解 $\boldsymbol{A}(L)$ は積分操作を通じて得られるため、この不連続性は滑らかされ、最終状態 $\boldsymbol{A}(L)$ は $\{z_k\}$ に関して微分可能です。(1)で示した勾配の存在がそれを示しています。目的関数 $G$ が $\boldsymbol{A}(L)$ の滑らかな関数であれば、合成関数 $J(\{z_k\})$ も滑らかであり、勾配法の適用が理論的に保証されます。

#### (3) 劣勾配（subgradient）の概念は必要か。標準的な勾配の枠組みで十分か。

本問題の最適化においては、標準的な勾配の枠組みで十分であり、劣勾配の概念は必要ありません。

**判断の根拠となる数学的条件と仮定:**

劣勾配は、目的関数が微分不可能な点を持つ場合に必要となります。目的関数の滑らかさは、以下の標準的な仮定に基づきます。

1.  **ダイナミクスの滑らかさ:** 結合波方程式の右辺 $\boldsymbol{f}$ が、状態変数 $\boldsymbol{A}$ に関して滑らかであること。
2.  **目的関数の滑らかさ:** 目的関数 $G$ が、終端状態 $\boldsymbol{A}(L)$ に関して滑らかであること（例：変換効率 $|A_3(L)|^2$）。

(2)で論じたように、これらの仮定の下で、目的関数 $J(\{z_k\})$ は連続的微分可能です。

**境界条件（ドメインの消滅）に関する考察:**
最適化過程でドメインが消滅する（$z_k = z_{k+1}$）場合、パラメータ空間の境界に到達します。しかし、ドメイン長がゼロに近づくと、そのドメインの伝播演算子 $\mathcal{P}_k$ は恒等写像に滑らかに近づくため、目的関数の微分可能性は境界上でも維持されます。

結論として、与えられた標準的な問題設定の下では、目的関数は滑らかであり、標準的な勾配の枠組みで十分対応可能です。