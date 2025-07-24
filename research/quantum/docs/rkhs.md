## 再生核ヒルベルト空間（RKHS）と関連作用素の要点

このドキュメントは、再生核ヒルベルト空間（RKHS）、そのカーネル関数、関連する積分作用素、そしてそれらの間の数学的な関係性についての要点をまとめたものです。

---

### 1. RKHSの定義と再生核

#### 再生核ヒルベルト空間 (Reproducing Kernel Hilbert Space, RKHS)
RKHSとは、関数のなすヒルベルト空間 $\mathcal{H}$ であり、任意の点 $x$ において、関数値を取り出す**評価汎関数** $\delta_x(f) = f(x)$ が連続（有界）となる空間です。

#### 再生核 (Reproducing Kernel)
評価汎関数 $\delta_x$ が連続であるため、**リースの表現定理**により、各点 $x$ に対してヒルベルト空間 $\mathcal{H}$ の元 $k_x$ が一意に存在し、以下の関係が成り立ちます。
$$f(x) = \langle f, k_x \rangle_{\mathcal{H}}$$
この $k_x$ を用いて、**再生核** $K(x, y)$ を以下のように定義します。
$$K(x, y) := \langle k_y, k_x \rangle_{\mathcal{H}}$$
この定義から、$k_x$ は関数 $K(x, \cdot)$ と同一視でき、RKHSの最も重要な性質である**再生性**が導かれます。
$$f(x) = \langle f, K(x, \cdot) \rangle_{\mathcal{H}}$$

---

### 2. カーネルの性質

#### 正定値性 (Positive Definiteness)
ある関数 $K(x, y)$ が何らかのRKHSの再生核であるための必要十分条件は、その関数が**正定値カーネル**であることです（**ムーア・アロンシャインの定理**）。

正定値カーネルは、任意の点の組 $\{x_i\}_{i=1}^n$ と任意の複素数列 $\{c_i\}_{i=1}^n$ に対して、以下の条件を満たします。
$$\sum_{i=1}^n \sum_{j=1}^n c_i \overline{c_j} K(x_i, x_j) \ge 0$$
この性質により、グラム行列 $[K(x_i, x_j)]$ は半正定値エルミート行列となり、特に対角成分は $K(x, x) = \|k_x\|_{\mathcal{H}}^2 \ge 0$ となります。

---

### 3. 積分作用素とスペクトル分解

#### 積分作用素
再生核 $K(x, y)$ は、$L^2$ 空間上の**積分作用素** $T$ を以下のように定義します。
$$(Tf)(x) = \int K(x, y) f(y) dy$$

#### 作用素の性質とスペクトル定理
カーネル $K$ が連続、エルミート対称、かつ正定値である場合、作用素 $T$ は $L^2$ 空間上で**コンパクト**かつ**自己共役**な正定値作用素となります。
**スペクトル定理**により、$T$ は正の固有値列 $\{\lambda_k\}$ と、それに対応する正規直交固有関数系 $\{\phi_k\}$ を持ちます。
$$T\phi_k = \lambda_k \phi_k \quad (\lambda_k > 0, \lambda_k \to 0)$$

#### マーサーの定理 (Mercer's Theorem)
上記の条件下で、カーネル $K$ 自身も以下のようにスペクトル分解できます。
$$K(x, y) = \sum_{k=1}^{\infty} \lambda_k \phi_k(x) \overline{\phi_k(y)}$$

---

### 4. 空間と内積の関係

#### 内積の表現
RKHS $\mathcal{H}$ の内積は、積分作用素 $T$ の固有値と固有関数を用いて、$L^2$ 内積と以下のように関連付けられます。
$$\langle f, g \rangle_{\mathcal{H}} = \sum_{k=1}^{\infty} \frac{1}{\lambda_k} \langle f, \phi_k \rangle_{L^2} \overline{\langle g, \phi_k \rangle_{L^2}}$$

#### 空間の関係性
この内積の定義から、$\|f\|_{\mathcal{H}}^2 < \infty$ であるためには、$L^2$ におけるフーリエ係数 $\langle f, \phi_k \rangle_{L^2}$ が、$\sqrt{\lambda_k}$ よりも速くゼロに収束する必要があります。これは $\|f\|_{L^2}^2 < \infty$ よりも厳しい条件であり、$\mathcal{H}$ が $L^2$ の部分空間（より滑らかな関数の集まり）であることを意味します。

---

### 5. 主要な恒等式

RKHSの内積、$L^2$内積、そして積分作用素 $T$ の間には、以下の非常に重要な関係式が成り立ちます。

#### 内積の架け橋
$f, g \in \mathcal{H}$ である関数に対して、
$$\langle Tf, g \rangle_{\mathcal{H}} = \langle f, g \rangle_{L^2}$$

この恒等式から、作用素 $T$ がRKHSの内積 $\langle \cdot, \cdot \rangle_{\mathcal{H}}$ に関しても自己共役であるという、以下の性質が導かれます。
$$\langle Tf, g \rangle_{\mathcal{H}} = \langle f, Tg \rangle_{\mathcal{H}}$$

## まとめ
RKHSは、カーネル積分作用素の固有関数を正規直交基底とし、対応する固有値の逆数倍で伸縮した関数空間
