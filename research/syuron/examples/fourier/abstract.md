### 広帯域波長変換デバイスの設計：関数空間上の最適化問題としての定式化

#### 1. 背景と目的

本研究は、一次元（1D）分極反転構造を用いた広帯域波長変換デバイスの最適設計を目的とする。目標とする仕様は、中心波長 $\lambda_c = 1030 \text{nm}$ の周辺で、可能な限り広く平坦な変換効率スペクトルを得ることである。

この波長に応答させる最も単純な構造は、周期が約 $p \approx 7.2 \mu\text{m}$ の一様な分極反転格子である。しかし、このような基準構造の応答は本質的に狭帯域であり、目標を達成できない。したがって、この単純な周期構造から逸脱した、高度に非周期的な構造をいかに設計するかが問題の核心となる。

また、物理的に製造可能な設計は、以下の作製上の制約を満たさなければならない。
*   **デバイス全長:** $L = 10 \text{ mm}$
*   **最小反転幅:** $w_{min} = 3 \mu\text{m}$
*   **制御解像度:** $\Delta w = 50 \text{nm}$

#### 2. 問題の基本定式化

##### 2.1 設計空間と目的関数
デバイスの分極反転構造を、ヒルベルト空間 $\mathcal{H} = L^2([0, L])$ に属する実数値関数 $g(z)$ として定義する。分極が二つの状態のみを取るという物理的要請から、この関数は以下の振幅制約を満たす。
$$
|g(z)| = 1 \quad (\text{for a.e. } z \in [0, L])
$$
したがって、探索空間は $\mathcal{H}$ 全体ではなく、この制約を満たす関数の部分集合 $S \subset \mathcal{H}$ である。

変換効率スペクトルは、$g(z)$ のフーリエ変換 $G(k)$ を用いて $|G(k)|^2$ に比例する。目的関数 $J: S \to \mathbb{R}_{\ge 0}$ は、中心波長 $\lambda_c$ を含む関心領域（ROI）内のスペクトル形状に基づき、以下の性能指標の積として定義される。

1.  **最大変換効率 ($P_{max}$):** ROI内におけるスペクトルの最大値。
    $$
    P_{max}(g) = \max_{k \in \text{ROI}} |G(k; g)|^2
    $$
2.  **フラットトップ帯域幅 ($W_{flat}$):** ROI内において、効率が $P_{max}$ の95%以上であるスペクトル領域の幅。
    $$
    W_{flat}(g) = \text{Width of } \{ k \in \text{ROI} \mid |G(k; g)|^2 \ge 0.95 \cdot P_{max}(g) \}
    $$
3.  **目的関数:**
    $$
    J(g) = P_{max}(g) \cdot W_{flat}(g)
    $$

##### 2.2 最適化における困難性
作製可能な構造は、区分的に定数（矩形波）でなければならない。この制約と、前述の非凸な振幅制約 $|g(z)|=1$ は、最適化問題に以下の根本的な困難をもたらす。
1.  **組み合わせ性:** 区分定数関数の区間分割パターンを最適化する、組み合わせ最適化問題となる。
2.  **微分不可能性:** 関数の不連続性と非凸な制約により、勾配ベースの最適化手法を直接適用することができない。

#### 3. 提案アプローチ：階層的最適化と微分可能プログラミングによる解決

上記の困難性を克服するため、元々の組み合わせ最適化問題を、**階層的な連続最適化問題**として再定式化する。このアプローチは、探索空間の構造化と、勾配法を適用するための微分可能性の確保という二つの柱から成る。

##### 3.1 階層的最適化戦略：多様体の探索と多様体上の探索

本アプローチでは、探索空間を直接扱うのではなく、まず低次元パラメータ $\boldsymbol{\theta} \in \Theta$ によって規定される**部分多様体**の族 $\{M(\boldsymbol{\theta})\}_{\boldsymbol{\theta} \in \Theta}$ を定義する。各多様体 $M(\boldsymbol{\theta})$ は、特定の設計コンセプトや特徴を持つ構造関数の集合を表す。

この定式化に基づき、最適化プロセスを以下の**二段階（階層的）戦略**に分解する。

1.  **外部探索（多様体の選択）:** 各多様体上で達成可能な目的関数の最大値、すなわち最適値関数 $J^*(\boldsymbol{\theta}) = \sup_{g \in M(\boldsymbol{\theta})} J(g)$ を最大化するような、最適なパラメータ $\boldsymbol{\theta}^* \in \Theta$ を探索する。この「最良の多様体」を見つけ出す大域的な探索は、メタヒューリスティックやAIベースのアルゴリズムによって実行される。

2.  **内部最適化（多様体上の探索）:** 外部探索で選択された特定の多様体 $M(\boldsymbol{\theta})$ 上で、目的関数 $J(g)$ を最大化する最適な構造 $g^* \in M(\boldsymbol{\theta})$ を見つけ出す。

この階層的アプローチにより、広大な設計空間の探索を、より構造化され扱いやすい問題へと分解することができる。

##### 3.2 勾配法適用のための微分可能性の担保

上記戦略、特に高速な内部最適化を実現するためには、勾配ベースの手法が不可欠である。しかし、Section 2.2で述べた通り、元の問題設定は微分不可能である。そこで、問題全体を**エンドツーエンドで微分可能**になるように再設計する。これは、以下の二つの補題を解決することで達成される。

*   **補題1：微分可能な構造生成マップの導入。**
    各多様体 $M(\boldsymbol{\theta})$ を、パラメータから構造への**微分可能な構造生成マップ $\Phi$** によって定義する。$\Phi$ がその引数（多様体を規定する $\boldsymbol{\theta}$ や、多様体上の座標を定める $u$）に対して微分可能であるように設計する限り、パラメータから物理構造への写像は微分可能となる。

*   **補題2：微分可能な代理目的関数の導入。**
    元の目的関数 $J(g)$ に内在する `max` や `width` といった微分不可能な演算子を、その挙動を忠実に近似する**滑らかな代理関数 $J_{smooth}(g)$** に置き換える。

これら二つの補題が満たされることで、パラメータから最終評価値までの全計算プロセスが微分可能となる。その結果、内部最適化はリーマン多様体上の勾配法によって高速に実行可能となり、勾配計算自体も随伴法（Adjoint Method）によって効率化できる。このアプローチは、元々の組み合わせ最適化問題を、微分幾何学と微分可能プログラミングの枠組みで解くことを可能にする。

### 補題1の具体化：微分可能な構造生成マップ $\Phi$ の構築

物理的な分極反転構造 $g(z)$ は、値が $+1$ と $-1$ のみを取る区分定数関数です。これをパラメータから直接生成しようとすると、反転位置の微小な変化が関数の形を不連続に変えるため、微分が不可能になります。

この問題を解決するため、**連続緩和（Continuous Relaxation）**というアプローチを取ります。これは、最適化の途中では厳密な制約を少し緩めた滑らかな関数を扱い、最適化の最終段階で物理的に実現可能な区分定数関数に収束させる手法です。

#### Step 1: 連続補助関数 $\tilde{g}(z)$ の導入

まず、直接 $g(z)$ を設計するのではなく、その原型となる滑らかな**連続補助関数 $\tilde{g}(z; \boldsymbol{u})$** を定義します。この関数は、Bスプラインやフーリエ級数など、滑らかな基底関数 $\{\phi_j(z)\}_{j=1}^N$ の線形結合で表現します。

$$
\tilde{g}(z; \boldsymbol{u}) = \sum_{j=1}^N u_j \phi_j(z)
$$

ここで、$\boldsymbol{u} = (u_1, \dots, u_N)$ が多様体上の座標を定める最適化パラメータとなります。$\tilde{g}(z)$ は連続かつパラメータ $\boldsymbol{u}$ に対して微分可能です。

#### Step 2: 微分可能な射影（Projection）

次に、この連続補助関数 $\tilde{g}(z)$ を、振幅制約 $|g(z)| \approx 1$ を満たす関数 $g(z)$ へと「射影」します。理想的な射影は `sign` 関数（$\text{sign}(x)$）ですが、これは原点で微分不可能です。そこで、`sign` 関数を滑らかに近似する**ハイパボリックタンジェント（tanh）関数**を用います。

$$
g(z; \boldsymbol{u}, \beta) = \tanh(\beta \cdot \tilde{g}(z; \boldsymbol{u}))
$$

この写像は、任意のパラメータ $\boldsymbol{u}$ と $\beta > 0$ に対して微分可能です。

#### Step 3: アニーリングによる物理制約への収束

ハイパーパラメータ $\beta$ は、`tanh` 関数の「急峻さ」を制御します。
*   $\beta$ が小さいとき、`tanh` は滑らかなシグモイド曲線となり、$g(z)$ は $-1$ から $+1$ までの連続的な値を取り得ます。これにより、勾配法が機能しやすい滑らかなランドスケープが形成されます。
*   $\beta$ が大きいとき、`tanh` は `sign` 関数に漸近し、$g(z)$ はほぼ $+1$ と $-1$ の値のみを取る、物理的に意味のあるバイナリ構造に近づきます。

最適化の過程で $\beta$ を徐々に大きくしていく**アニーリング戦略**を採用することで、探索の初期段階では広域的な探索を可能にし、最終的には物理制約を満たす解へと収束させることができます。

この一連のプロセスにより、最適化パラメータ $\boldsymbol{u}$ から物理構造 $g(z)$ への**微分可能な構造生成マップ $\Phi$** が構築されます。

---

### 補題2の具体化：微分可能な代理目的関数 $J_{smooth}$ の構築

元の目的関数 $J(g) = P_{max}(g) \cdot W_{flat}(g)$ に含まれる `max` と `width` 演算子は、微分不可能な点を持ちます。これらを滑らかな関数で近似し、目的関数全体を微分可能にします。

#### 1. `max` 演算子の平滑化：Softmax（LogSumExp）

ROI内の離散化された波数点 $\{k_i\}_{i=1}^M$ におけるスペクトル強度を $P_i(g) = |G(k_i; g)|^2$ とします。このとき、最大値 $P_{max}$ は、**LogSumExp (LSE) 関数** を用いて次のように滑らかに近似できます。

$$
P_{max}(g) = \max_{i} P_i(g) \quad \approx \quad P_{max, smooth}(g; \gamma) = \frac{1}{\gamma} \log \left( \sum_{i=1}^M \exp(\gamma P_i(g)) \right)
$$

この関数は「Softmax」とも呼ばれ、解析的に微分可能です。ハイパーパラメータ $\gamma > 0$ が大きいほど、真の `max` 関数への近似精度が高まります。

#### 2. `width` 演算子の平滑化：シグモイド関数による近似

フラットトップ帯域幅 $W_{flat}$ は、効率がある閾値（例: $0.95 \cdot P_{max}$）を超える領域の幅として定義されます。この「閾値処理」は本質的にステップ関数であり、微分不可能です。

この問題を解決するため、各波数点 $k_i$ が帯域幅の条件を満たすかどうかを判定するインジケータ関数を、滑らかな**シグモイド関数 $\sigma(x; \alpha) = 1 / (1 + e^{-\alpha x})$** で近似します。

まず、条件式を $P_i(g) - 0.95 \cdot P_{max, smooth}(g) \ge 0$ と変形します。この式の左辺をシグモイド関数の入力とします。

$$
\tilde{H}_i(g; \alpha) = \sigma(P_i(g) - 0.95 \cdot P_{max, smooth}(g); \alpha)
$$

$\tilde{H}_i$ は、$P_i$ が閾値を大きく超えると $1$ に近づき、大きく下回ると $0$ に近づく滑らかな関数です。
これを用いて、帯域幅を近似的に計算します（$\Delta k$ は波数点のサンプリング間隔）。

$$
W_{flat, smooth}(g; \gamma, \alpha) = \Delta k \sum_{i=1}^M \tilde{H}_i(g; \alpha)
$$

ハイパーパラメータ $\alpha > 0$ を大きくするほど、この近似は元のステップ関数的な判定に近づきます。

#### 3. 代理目的関数の完成

以上を統合し、エンドツーエンドで微分可能な**代理目的関数 $J_{smooth}$** を定義します。

$$
J_{smooth}(g; \gamma, \alpha) = P_{max, smooth}(g; \gamma) \cdot W_{flat, smooth}(g; \gamma, \alpha)
$$

この $J_{smooth}$ は、その引数である構造 $g$ を通じて、最終的には構造生成パラメータ $\boldsymbol{u}$ の微分可能な関数となります。

### まとめと次への展望

補題1と2により、元の組み合わせ最適化問題は、以下の特徴を持つ連続最適化問題に変換されました。

*   **探索空間:** 最適化パラメータ $\boldsymbol{u}$ がなすユークリッド空間（またはリーマン多様体）。
*   **構造生成:** パラメータ $\boldsymbol{u}$ から物理構造 $g(z)$ への写像は、`tanh` 関数により微分可能。
*   **目的関数:** 目的関数 $J_{smooth}$ は、Softmaxとシグモイド関数により微分可能。

これにより、パラメータ $\boldsymbol{u}$ に関する目的関数の勾配 $\nabla_{\boldsymbol{u}} J_{smooth}$ を、連鎖律（Chain Rule）を用いて解析的に計算できます。特に、パラメータ数 $N$ が大きい場合でも、**随伴法（Adjoint Method）**を用いることで、一回の順伝播・逆伝播計算で効率的に全勾配を得ることが可能です。

この微分可能な枠組みが整ったことで、強力な勾配法（例：リーマン多様体上の共役勾配法やAdam）を内部最適化（多様体上の探索）に適用し、高速かつ高精度なデバイス設計が実現可能となります。

### その他のアイデア

#### アイデア1：ペナルティ関数による制約の組み込み

このアプローチは、目的関数に**制約違反の度合いに応じたペナルティ項**を導入し、最適化の過程で解を許容領域へとソフトに誘導するものである。目的関数を以下のように修正する。

$$
J_{penalized}(g) = J_{smooth}(g) - \lambda_w P_{width}(g) - \lambda_{q} P_{quantize}(g)
$$

*   **最小反転幅ペナルティ $P_{width}(g)$:** 構造 $g(z)$ から反転領域の幅を（微分可能な形で）検出し、$w_{min}$ を下回る領域が存在する場合にその違反量に応じたペナルティを課す関数。
*   **量子化ペナルティ $P_{quantize}(g)$:** 関数 $g(z)$ が $+1$ または $-1$ からどれだけ離れているか（例：$\int (g(z)^2-1)^2 dz$）や、空間的に平坦でないか（例：$\int (g'(z))^2 dz$）を測定し、ペナルティを課す関数。

これらのペナルティ項を微分可能に設計することで、最適化アルゴリズムは性能を最大化しつつ、同時に物理的に望ましい構造へと自律的に収束していくことが期待される。これは、既存のフレームワークに比較的容易に追加実装できる、現実的な拡張策である。

#### アイデア2：制約を内包する再パラメータ化

より根本的な解決策として、**最適化に用いるパラメータの定義そのものを工夫し、生成される構造が本質的に制約を満たすように設計する**アプローチが考えられる。

例えば、構造を反転位置のリストではなく、**各反転区間の幅のリスト $\{d_1, d_2, \dots, d_K\}$** で表現する。そして、この幅 $d_k$ を、制約を内包するような新たな最適化パラメータ $v_k$ を用いて以下のように定義する。

$$
d_k = w_{min} + \text{round} \left( \frac{\text{softplus}(v_k)}{\Delta w} \right) \cdot \Delta w
$$

*   `softplus(v_k) = log(1+e^{v_k})` は常に正の値を取る滑らかな関数であり、$d_k$ が $w_{min}$ を下回ることを防ぐ。
*   `round` 関数と `Δw` により、制御解像度の制約が反映される。（`round` は微分不可能だが、Straight-Through Estimatorなどのテクニックで勾配計算を近似可能）

この「生成的な」アプローチは、そもそも探索空間を物理的に実現可能な構造の集合に限定するため、ペナルティ法よりも直接的かつ強力な解法となりうる。ただし、パラメータ化の設計自体がより高度な数学的考察を要する。

### キーワード
* optimize-then-discretize
* 劣勾配
