### **研究目的**
SHG（第二高調波発生）とSFG（和周波発生）によるカスケード二次非線形過程を用いたTHG（第三高調波発生）のスペクトル分布設計。

### **カスケードTHGを記述する結合波方程式**

本研究のシミュレーションは、カスケード二次非線形過程における三つの波（基本波、第二高調波、第三高調波）の相互作用を記述する結合波方程式に基づきます。以下に、理論的な出発点から数値計算に用いる形式までを段階的に示します。

#### **1. SVEAに基づく基本方程式 ($\chi^{(2)}$)**

Slowly Varying Envelope Approximation (SVEA) を用い、非線形感受率 $\chi^{(2)}(z)$ で直接記述した基本方程式は以下の通りです。これは多くの理論的な導出における出発点となります。

$$\frac{\partial E_1}{\partial z} = i \frac{\omega_1^2}{2k_1 c^2} \chi^{(2)}(z) [E_2 E_1^* \exp(i\Delta k_1 z) + E_3 E_2^* \exp(i\Delta k_2 z)]$$

$$\frac{\partial E_2}{\partial z} = i \frac{\omega_2^2}{2k_2 c^2} \chi^{(2)}(z) [\frac{1}{2} E_1^2 \exp(-i\Delta k_1 z) + E_3 E_1^* \exp(i\Delta k_2 z)]$$

$$\frac{\partial E_3}{\partial z} = i \frac{\omega_3^2}{2k_3 c^2} \chi^{(2)}(z) [E_1 E_2 \exp(-i\Delta k_2 z)]$$

*補足: ここでは係数部分を $k_{j0} = \omega_j/c$, $k_j = n_j k_{j0}$ の関係を用いて $\frac{k_{j0}^2}{2k_j} = \frac{(\omega_j/c)^2}{2 n_j (\omega_j/c)} = \frac{\omega_j}{2n_j c}$ ではなく、原義に近い形で表記しています。*

#### **2. 実用的な一般形 ($d_{eff}$)**

実用上、非線形効果の大きさは実効非線形係数 $d_{eff}$ で表されることが多く、$\chi^{(2)} = 2d_{eff}$ の関係を用いて上記の式を書き換えると、以下の物理的に直感的な形式が得られます。

$$\frac{\partial E_1}{\partial z} = i \frac{\omega_1 d_{eff}(z)}{n_1 c} \left[ E_2 E_1^* \exp(i\Delta k_1 z) + E_3 E_2^* \exp(i\Delta k_2 z) \right]$$

$$\frac{\partial E_2}{\partial z} = i \frac{\omega_2 d_{eff}(z)}{n_2 c} \left[ \frac{1}{2} E_1^2 \exp(-i\Delta k_1 z) + E_3 E_1^* \exp(i\Delta k_2 z) \right]$$

$$\frac{\partial E_3}{\partial z} = i \frac{\omega_3 d_{eff}(z)}{n_3 c} \left[ E_1 E_2 \exp(-i\Delta k_2 z) \right]$$

#### **3. 数値計算のための規格化形式**

数値シミュレーションを効率的に行うため、光強度に比例する複素振幅 $A_j$ と、相互作用の強さを表す結合係数 $\kappa(z)$ を導入し、近似6を行い方程式を規格化します。

$$\frac{\partial A_1}{\partial z} = i \kappa(z) \left[ A_2 A_1^* e^{i\Delta k_1 z} + A_3 A_2^* e^{i\Delta k_2 z} \right]$$

$$\frac{\partial A_2}{\partial z} = i \, 2\kappa(z) \left[ \frac{1}{2} A_1^2 e^{-i\Delta k_1 z} + A_3 A_1^* e^{i\Delta k_2 z} \right]$$

$$\frac{\partial A_3}{\partial z} = i \, 3\kappa(z) \left[ A_1 A_2 e^{-i\Delta k_2 z} \right]$$

---

### **主要な変数とパラメータの定義**

* **複素振幅 $A_j(z)$**:
    * $j=1, 2, 3$ はそれぞれ基本波(FW)、第二高調波(SHW)、第三高調波(THW)に対応します。
    * 光強度 $I_j$ は $I_j(z) = |A_j(z)|^2$ で与えられます。

* **結合係数 $\kappa(z)$**:
    * $\kappa(z) \equiv d_{eff}(z) \omega_1 \sqrt{\frac{2}{n^3 c^3 \epsilon_0}}$
    * $d_{eff}(z)$: 実効非線形光学係数。媒質の分極反転構造などに依存します。
    * $\omega_1$: 基本波の角周波数。
    * $n$: 媒質の代表的な屈折率（近似値）。
    * $c, \epsilon_0$: 真空中の光速と真空の誘電率。

* **位相不整合 $\Delta k_j$**:
    * $\Delta k_1 = k_2 - 2k_1$: SHG過程の位相不整合。
    * $\Delta k_2 = k_3 - k_2 - k_1$: SFG過程の位相不整合。
    * **重要**: 位相整合条件はエネルギー変換効率を決定するため、波数 $k_j = n_j(\omega_j)\omega_j/c$ の計算では、材料の屈折率の波長分散 $n_j(\omega_j)$ を厳密に考慮する必要があります。

---

### **方程式の導出における主要な近似**

上記の方程式は、以下の標準的な近似に基づいています。

1.  **SVEA (Slowly Varying Envelope Approximation)**: 光波の振幅包絡線は、光の1波長・1周期のスケールに比べてゆっくり変化すると仮定します。
2.  **平面波近似**: ビームの回折効果を無視し、光を横方向に無限に広がる平面波として扱います。
3.  **CW (連続波) または準CW近似**: パルス幅が十分に長く、群速度分散（GVD）などによるパルスの時間的な形状変化を無視します。
4.  **無損失媒質**: 媒質による光の線形吸収や散乱を無視します。
5.  **同一直線伝播**: 相互作用する全ての光波が完全に平行に伝播すると仮定し、ビームウォークオフなどを無視します。
6.  **係数における近似**: 振幅の結合係数を導出する際、屈折率 $n_j$の周波数依存性を無視し、代表値で近似します（$n_j \approx n$）。ただし、位相不整合 $\Delta k$ の計算では分散を厳密に扱います。
