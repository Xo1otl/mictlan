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

# NPDA

$$
A_{2\omega,out}=\sum^{N-1}_{j=0}\Delta A^{(k)}_{2\omega}
$$

$$
\Delta A^{(k)}_{2\omega} = (-j\kappa_d^{(k)}\Delta A_{in}^2 L_d^{(k)} e^{j\Delta L_d^{(k)}}\cdot \frac {sin(\Delta L_d^{(k)})} {\Delta L_d^{(k)}}) \cdot e^{j\Phi(z_k)}
$$

* $L_d^{(k)}$: k番目のドメイン幅
* $\kappa_d^{(k)}$: k番目のドメインの非線形結合係数
* $z_k$: k番目のドメインの開始点のz座標

実は、この計算がちゃんと動作してる原理がいまいちわかってないから解説がほしいです

式の最後で$e^{j\Phi (z_k)}$をかけると計算がうまく行く理由がわからん

基本波がSH波を発生させるのは分かる

発生源である基本波の位相がドメインごとに異なっており、その補正を行ってるってこと？

ご質問ありがとうございます。NPDA（非枯渇ポンプ近似）におけるSH波の振幅計算、特に位相因子 $e^{j\Phi(z_k)}$ の役割について解説します。

与えられた前提条件とNPDAの式を基に、順を追って説明します。

**1. NPDAにおけるSH波振幅の基本式**

SH波の振幅 $A^{2\omega}(z)$ に関する非線形結合モード方程式 (2.59b) は以下のように与えられています。
$$\frac {d}{dz} A^{2\omega}(z) = -j\boldsymbol\kappa(z)[A^\omega(z)]^2\exp(+j(2\Delta)z) \quad (*1)$$
ここで、$2\Delta = \beta^{2\omega} - 2\beta^\omega$ は位相不整合の大きさを表します。

NPDAでは、基本波の振幅 $A^\omega(z)$ はSH波への変換によって減衰しないと仮定し、$A^\omega(z) \approx A_{in}$（入力基本波振幅、実数と仮定）とします。
また、SH波の初期振幅は $A^{2\omega}(0) = 0$ とします。
このとき、デバイス長 $L$ でのSH波の振幅は、式(*1)を $z=0$ から $L$ まで積分することで得られます。
$$A^{2\omega}(L) = \int_0^L -j\kappa(z') [A_{in}]^2 \exp(+j2\Delta z') dz'$$
QPMデバイスでは、非線形結合係数 $\kappa(z')$ はドメインごとに $\kappa_d^{(k)}$ という一定値を持ちます。デバイスが $N$ 個のドメインから構成され、$k$ 番目のドメインが $z_k$ から $z_{k+1}$ (ここで $z_{k+1} = z_k + L_d^{(k)}$、$L_d^{(k)}$ は $k$ 番目のドメイン幅) まで存在するとすると、積分は各ドメインの寄与の和として書けます。
$$A^{2\omega}(L) = -j A_{in}^2 \sum_{k=0}^{N-1} \int_{z_k}^{z_{k+1}} \kappa_d^{(k)} \exp(+j2\Delta z') dz' \quad (*2)$$
ここで、$\kappa_d^{(k)}$ は $k$ 番目のドメインにおける $\kappa(z)$ の値で、$\kappa_d^{(k)} = \kappa_{mag}(-1)^{n(z_k)}$ です。

**2. 各ドメインからのSH波生成量の導出**

各ドメインでの積分を実行します。
$$\int_{z_k}^{z_{k+1}} \exp(+j2\Delta z') dz' = \left[ \frac{\exp(+j2\Delta z')}{j2\Delta} \right]_{z_k}^{z_{k+1}}$$
$$= \frac{1}{j2\Delta} (\exp(+j2\Delta z_{k+1}) - \exp(+j2\Delta z_k))$$
$z_{k+1} = z_k + L_d^{(k)}$ を代入すると、
$$= \frac{1}{j2\Delta} (\exp(+j2\Delta (z_k+L_d^{(k)})) - \exp(+j2\Delta z_k))$$
$$= \frac{1}{j2\Delta} \exp(+j2\Delta z_k) (\exp(+j2\Delta L_d^{(k)}) - 1)$$
ここで、$\exp(jX) - 1 = e^{jX/2}(e^{jX/2} - e^{-jX/2}) = e^{jX/2} 2j\sin(X/2)$ の関係を用いると、$X = 2\Delta L_d^{(k)}$ なので、
$$\exp(+j2\Delta L_d^{(k)}) - 1 = e^{j\Delta L_d^{(k)}} 2j\sin(\Delta L_d^{(k)})$$
よって、積分部分は、
$$
\frac{1}{j2\Delta} \exp(+j2\Delta z_k) e^{j\Delta L_d^{(k)}} 2j\sin(\Delta L_d^{(k)})
= L_d^{(k)} \frac{\sin(\Delta L_d^{(k)})}{\Delta L_d^{(k)}} e^{j\Delta L_d^{(k)}} \exp(+j2\Delta z_k)
$$

これを式(*2)の総和の中に戻すと、$A^{2\omega}(L)$ は各ドメインからの寄与 $\Delta A^{(k)}_{2\omega}$ の和として書けます。
$$
A^{2\omega}(L) = \sum_{k=0}^{N-1} \left( -j \kappa_d^{(k)} A_{in}^2 L_d^{(k)} \frac{\sin(\Delta L_d^{(k)})}{\Delta L_d^{(k)}} e^{j\Delta L_d^{(k)}} \right) \exp(+j2\Delta z_k)
$$
これが、ご提示のNPDAの式
$$
A_{2\omega,out} = \sum^{N-1}_{k=0}\Delta A^{(k)}_{2\omega}
$$
$$
\Delta A^{(k)}_{2\omega} = \left(-j\kappa_d^{(k)}A_{in}^2 L_d^{(k)} e^{j\Delta L_d^{(k)}}\cdot \frac {\sin(\Delta L_d^{(k)})} {\Delta L_d^{(k)}}\right) \cdot e^{j\Phi(z_k)}
$$
と一致します。
比較すると、位相因子 $e^{j\Phi(z_k)}$ は
$$e^{j\Phi(z_k)} = \exp(+j2\Delta z_k)$$
すなわち、
$$\Phi(z_k) = 2\Delta z_k = (\beta^{2\omega} - 2\beta^\omega)z_k$$
であることがわかります。

**3. $e^{j\Phi(z_k)} = e^{j2\Delta z_k}$ の物理的意味**

この位相因子 $e^{j2\Delta z_k}$ は、**$k$ 番目のドメインが結晶の開始点 ($z=0$) から $z_k$ の位置にあることによる、SHGプロセスにおける初期位相のずれ**を表しています。

* **位相不整合の影響:** 位相不整合 $2\Delta$ が存在すると、基本波とSH波の間の位相関係は伝播距離と共に $2\Delta z$ のように変化していきます。これは、SHGの効率に直接影響します。
* **各ドメインの開始位置:** $k$ 番目のドメインは $z=z_k$ から始まります。このドメイン内でSH波が生成される際、その「開始地点」までに蓄積された位相のずれ $2\Delta z_k$ が、そのドメインで生成されるSH波の初期位相に影響します。
* **コヒーレントな加算:** デバイス全体のSH波出力は、各ドメインで生成されたSH波のコヒーレントな和（位相を考慮した足し算）です。$e^{j2\Delta z_k}$ の因子は、各ドメインで生成されたSH波の成分を、共通の位相基準のもとで正しく足し合わせるために不可欠です。この因子がないと、各ドメインからの寄与の位相がバラバラになり、効率的なSHGは実現できません。

**ユーザーの疑問点について**

> 式の最後で $e^{j\Phi (z_k)}$ をかけると計算がうまく行く理由がわからん
> 基本波がSH波を発生させるのは分かる
> 発生源である基本波の位相がドメインごとに異なっており、その補正を行ってるってこと？

ご指摘の通り、これは非常に良い洞察です。より正確に表現すると以下のようになります。

* SH波は基本波から発生します。このとき、SHGの効率は基本波と発生するSH波の位相関係に強く依存します。
* 位相不整合 $2\Delta$ があるため、基本波のエネルギーがSH波に変換される際の「実効的な位相」は、伝播位置 $z$ と共に $e^{j2\Delta z}$ の形で変化します。
* $k$ 番目のドメインは $z_k$ から始まります。したがって、このドメインでSH波が生成されるプロセス全体には、$e^{j2\Delta z_k}$ という形で「そのドメインの開始位置に起因する位相オフセット」が乗じられます。
* $\Delta A^{(k)}_{2\omega}$ の最初の括弧内の部分は、もしドメインが $z=0$ から始まったと仮定した場合の、そのドメイン長 $L_d^{(k)}$ と位相不整合 $\Delta$ で決まるSH波の生成量（複素振幅）を表します。
* それに $e^{j\Phi(z_k)} = e^{j2\Delta z_k}$ を乗じることで、$k$ 番目のドメインが実際に $z_k$ の位置から始まることによる位相効果を正しく取り込み、全てのドメインからの寄与を共通の位相基準で評価して足し合わせることができるのです。

**QPM (擬似位相整合) との関係**

QPMでは、この $e^{j2\Delta z_k}$ による位相のずれを補償するように、非線形光学係数 $\kappa(z)$ の符号をドメインごとに反転させます ($\kappa_d^{(k)} = \kappa_{mag}(-1)^{n(z_k)}$)。
理想的なQPMでは、$\kappa_d^{(k)} \exp(+j2\Delta z_k)$ の項が、各ドメインでほぼ同じ位相を向くようにドメイン長 $L_d^{(k)}$ と符号反転が設計されます。これにより、各ドメインで生成されたSH波が建設的に干渉しあい、SH波の出力が効率的に増大します。

したがって、$e^{j\Phi(z_k)}$ は、QPMデバイスにおけるSHGを正しく記述し、各ドメインからの寄与をコヒーレントに重ね合わせるために本質的な役割を果たす位相補正項と言えます。
