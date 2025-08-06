# 研究目的
shgとsfgによるcascaded second-order nonlinear processによるthgのスペクトル分布の設計

## cascaded second-order nonlinear processによるthird harmonic generationをシミュレーションするためのcoupled mode equations

### SVEAを用いて導出されるequations
Assuming the standard definition of the complex electric field amplitude ($E_{real}(z,t) = \text{Re}[E(z)e^{i(kz-\omega t)}]$) and the nonlinear susceptibility $\chi^{(2)}$, the coupled wave equations is:

$$\frac{\partial E_1}{\partial z} = i \frac{k_{10}^2}{2k_1} \chi^{(2)}(z) [E_2 E_1^* \exp(i\Delta k_1 z) + E_3 E_2^* \exp(i\Delta k_2 z)]$$

$$\frac{\partial E_2}{\partial z} = i \frac{k_{20}^2}{2k_2} \chi^{(2)}(z) [\frac{1}{2} E_1^2 \exp(-i\Delta k_1 z) + E_3 E_1^* \exp(i\Delta k_2 z)]$$

$$\frac{\partial E_3}{\partial z} = i \frac{k_{30}^2}{2k_3} \chi^{(2)}(z) [E_1 E_2 \exp(-i\Delta k_2 z)]$$

Where $\Delta k_1 = k_2 - 2k_1$, $\Delta k_2 = k_3 - k_2 - k_1$, $k_{10} = \omega_1 / c$, $k_{20} = \omega_2 / c$, $k_{30} = \omega_3 / c$,
$k_1 = n_1 k_{10}$, $k_2 = n_2 k_{20}$, $k_3 = n_3 k_{30}$, $E_1 (E_2, E_3)$ and $n_1 (n_2, n_3)$ are respectively the electric field and refractive index for FW (SHW, THW), $c$ is light speed in vacuum, $\chi^{(2)} (z)$ is the second-order nonlinear susceptibility distribution of the CPPLN structure. 

### $d_{eff}$ を用いた結合波方程式

カスケード過程を扱う際、多くの教科書や論文で出発点として用いられる、実効非線形係数 $d_{eff}$ を用いた最も一般的な結合波方程式は以下の通りです。

$$\frac{\partial E_1}{\partial z} = i \frac{\omega_1 d_{eff}(z)}{n_1 c} \left[ E_2 E_1^* \exp(i\Delta k_1 z) + E_3 E_2^* \exp(i\Delta k_2 z) \right]$$

$$\frac{\partial E_2}{\partial z} = i \frac{\omega_2 d_{eff}(z)}{n_2 c} \left[ \frac{1}{2} E_1^2 \exp(-i\Delta k_1 z) + E_3 E_1^* \exp(i\Delta k_2 z) \right]$$

$$\frac{\partial E_3}{\partial z} = i \frac{\omega_3 d_{eff}(z)}{n_3 c} \left[ E_1 E_2 \exp(-i\Delta k_2 z) \right]$$

### 複素振幅と屈折率の近似と$\omega$の条件などから得られる結合波方程式

$$\frac{\partial A_1}{\partial z} = i \kappa(z) \left[ A_2 A_1^* e^{i\Delta k_1 z} + A_3 A_2^* e^{i\Delta k_2 z} \right]$$

$$\frac{\partial A_2}{\partial z} = i \, 2\kappa(z) \left[ \frac{1}{2} A_1^2 e^{-i\Delta k_1 z} + A_3 A_1^* e^{i\Delta k_2 z} \right]$$

$$\frac{\partial A_3}{\partial z} = i \, 3\kappa(z) \left[ A_1 A_2 e^{-i\Delta k_2 z} \right]$$

### 変数の定義

* $A_j(z)$: 複素振幅 (Complex Amplitude)
    * $j=1, 2, 3$ はそれぞれ基本波(FW)、第2高調波(SHW)、第3高調波(THW)に対応します。
    * 光強度 $I_j$ は $I_j(z) = |A_j(z)|^2$ で与えられます。

* $\kappa(z)$: 結合係数 (Coupling Coefficient)
    * $\kappa(z) \equiv d_{eff}(z) \omega_1 \sqrt{\frac{2}{n^3 c^3 \epsilon_0}}$
    * $d_{eff}(z)$: 実効非線形光学係数（媒質の構造に依存）。
    * $\omega_1$: 基本波の角周波数。
    * $n$: 媒質の代表的な屈折率（近似値）。
    * $c, \epsilon_0$: 真空中の光速と真空の誘電率。

* $\Delta k_j$: 位相不整合 (Phase Mismatch)
    * $\Delta k_1 = k_2 - 2k_1$: SHG過程の位相不整合。
    * $\Delta k_2 = k_3 - k_2 - k_1$: SFG過程の位相不整合。
    * 波数 $k_j = n_j(\omega_j)\omega_j/c$ の計算では、屈折率の波長分散を厳密に考慮した $n_j(\omega_j)$ を用います。

### **結合波方程式における主要な近似**

1.  **SVEA (Slowly Varying Envelope Approximation)**
    光波の振幅が、光の1波長や1周期のスケールではほとんど変化しないという近似。方程式を導出する上での根幹です。

2.  **CW (連続波) または準CW近似**
    パルス幅が長く、群速度分散（GVD）などパルスの時間的な形状変化を無視する近似。ご提示の方程式には時間微分項が含まれていません。

3.  **平面波近似 (Plane Wave Approximation)**
    ビームを横方向に無限に広がる平面波とみなし、回折やビームの空間プロファイル（ガウシアン形状など）を無視する近似です。

4.  **無損失媒質の仮定 (Lossless Medium)**
    結晶による線形吸収や散乱がないと仮定する近似。より現実に近いモデルでは、減衰項が追加されます。

5.  **同一直線伝播の仮定 (Collinear Propagation)**
    相互作用する全ての光波が完全に平行に進むとする近似。複屈折結晶でのビームウォークオフ効果などを無視します。

6.  **結合係数内における屈折率の近似**
    位相整合($\Delta k$)の計算では分散の厳密な考慮が必須ですが、振幅への影響は小さいため、結合係数の中では屈折率$n_j$を一定と見なす近似です。

以上の6つが、今回の方程式セットで使用されている近似です。
