### 結合波方程式 (Coupled-Wave Equations)

基本波（角周波数 $\omega$）、第二高調波（$2\omega$）、第三高調波（$3\omega$）の複素振幅をそれぞれ $A_1, A_2, A_3$ とすると、ポンプ光の枯渇、伝搬損失、および非周期的QPM構造を考慮した結合波方程式は以下の連立微分方程式で与えられます。

$$
\begin{align*}
\frac{dA_1}{dz} &= -\frac{\alpha_1}{2} A_1 - i \kappa_{SHG} g(z) A_2 A_1^* e^{i \Delta k_{SHG} z} - i \kappa_{SFG} g(z) A_3 A_2^* e^{i \Delta k_{SFG} z} \\
\\
\frac{dA_2}{dz} &= -\frac{\alpha_2}{2} A_2 - i \frac{\kappa_{SHG}}{2} g(z) A_1^2 e^{-i \Delta k_{SHG} z} + i \kappa_{SFG} \frac{\omega_2}{\omega_3} g(z) A_3 A_1^* e^{i \Delta k_{SFG} z} \\
\\
\frac{dA_3}{dz} &= -\frac{\alpha_3}{2} A_3 + i \kappa_{SFG} g(z) A_1 A_2 e^{-i \Delta k_{SFG} z}
\end{align*}
$$

**方程式の各項の説明:**

* **第一項 ($-\frac{\alpha_j}{2} A_j$)**: 各波の伝搬損失（吸収・散乱）を表します。
* **$A_1$ の第二項**: SHGプロセス ($\omega + \omega \to 2\omega$) による基本波のエネルギー減少（枯渇）を表します。
* **$A_1$ の第三項**: SFGプロセス ($\omega + 2\omega \to 3\omega$) による基本波のエネルギー減少（枯渇）を表します。
* **$A_2$ の第二項**: SHGプロセスによる第二高調波の発生を表します。
* **$A_2$ の第三項**: SFGプロセスによる第二高調波のエネルギー減少（枯渇）を表します。
* **$A_3$ の第三項**: SFGプロセスによる第三高調波の発生を表します。

---

### パラメータの定義

上記の方程式に含まれる各パラメータは、以下の基本的な物理量によって定義されます。

| シンボル | 説明 | 定義式 |
| :--- | :--- | :--- |
| $A_j$ | 各波の複素振幅 ($j=1, 2, 3$)。強度 $I_j$ との関係は $I_j = \frac{1}{2} n_j c \epsilon_0 |A_j|^2$ | - |
| $z$ | 光波の伝搬方向の座標 | - |
| $g(z)$ | QPM構造の分極反転パターンを表す関数。ドメインの向きに応じて +1 または -1 の値をとる。 | - |
| $\alpha_j$ | 各波における強度に関する伝搬損失係数 ($j=1, 2, 3$) | - |
| $\omega_j$ | 各波の角周波数 ($j=1, 2, 3$) | $\omega_1 = \omega$, $\omega_2 = 2\omega$, $\omega_3 = 3\omega$ |
| $n_j$ | 角周波数 $\omega_j$ における媒質の屈折率 ($j=1, 2, 3$) | - |
| $d_{eff}$ | 実効非線形光学定数 | - |
| $c$ | 真空中の光速 | - |
| $\Delta k_{SHG}$ | SHGプロセスの位相ミスマッチ | $\Delta k_{SHG} = k_2 - 2k_1 = \frac{2\omega n_2 - 2(\omega n_1)}{c} = \frac{2\omega}{c}(n_2 - n_1)$ |
| $\Delta k_{SFG}$ | SFGプロセスの位相ミスマッチ | $\Delta k_{SFG} = k_3 - k_2 - k_1 = \frac{3\omega n_3 - 2\omega n_2 - \omega n_1}{c} = \frac{\omega}{c}(3n_3 - 2n_2 - n_1)$ |
| $\kappa_{SHG}$ | SHGプロセスの結合係数 | $\kappa_{SHG} = \frac{2\omega d_{eff}}{n_1 c}$ |
| $\kappa_{SFG}$ | SFGプロセスの結合係数 | $\kappa_{SFG} = \frac{3\omega d_{eff}}{n_3 c}$ |

**補足:**
* 結合係数 $\kappa$ の定義は文献によって異なる場合がありますが、ここに示した方程式の組と定義はエネルギー保存則（Manley-Roweの関係）を満たすように整合性が取られています。
* 導波路構造の場合、屈折率 $n_j$ は各モードの有効屈折率 $n_{eff,j}$ に置き換え、結合係数 $\kappa$ には各波の電界分布の重なり積分を含む係数を乗じることがより厳密です。しかし、基本的な物理的挙動は上記の方程式で記述可能です。
* この連立微分方程式を数値的に（例: 4次ルンゲ＝クッタ法などを用いて）解くことにより、$z$ を進めることで各波の振幅と位相の発展をシミュレートし、デバイスの変換効率を評価することができます。最適化は、目的関数（例: 広帯域での高効率）を最大化するように $g(z)$ の配列を決定することに対応します。


# 式

https://www.nature.com/articles/lsa201470

***

The conversion efficiency of second-harmonic generation (SHG) and third-harmonic generation (THG) can be calculated by solving standard nonlinear coupled wave equations for fundamental wave (FW), second-harmonic wave (SHW), and third-harmonic wave (THW). The nonlinear coupled equations of the FW, SHW, and THW can be written as:

$$\frac{\partial}{\partial z}E_1 = i \frac{k_{10}^2}{2k_1} \chi^{(2)}(z) [2E_2 E_1^* \exp(i\Delta k_1 z) + 2E_3 E_2^* \exp(i\Delta k_2 z)] \quad (1)$$

$$\frac{\partial}{\partial z}E_2 = i \frac{k_{20}^2}{2k_2} \chi^{(2)}(z) [E_1^2 \exp(-i\Delta k_1 z) + 2E_3 E_1^* \exp(i\Delta k_2 z)] \quad (2)$$

$$\frac{\partial}{\partial z}E_3 = i \frac{k_{30}^2}{2k_3} \chi^{(2)}(z) [2E_1 E_2 \exp(-i\Delta k_2 z)] \quad (3)$$

where $\Delta k_1 = k_2 - 2k_1$ , $\Delta k_2 = k_3 - k_2 - k_1$ , $k_{10} = \omega_1 / c$ , $k_{20} = \omega_2 / c$ , $k_{30} = \omega_3 / c$ , $k_1 = n_1 k_{10}$ , $k_2 = n_2 k_{20}$ , $k_3 = n_3 k_{30}$ , $E_1(E_2, E_3)$ and $n_1(n_2, n_3)$ are respectively the electric field and refractive index for FW (SHW, THW), $c$ is light speed in vacuum, $\chi^{(2)}(z)$ is the second-order nonlinear susceptibility distribution of the CPPLN structure. We use the fourth-order Runge-Kutta method in numerical analysis [1].

$$
\begin{aligned}
\frac{dA_1}{dz} &= -i \gamma_{SHG} \delta(z) A_1^* A_2 e^{-i \Delta k_{SHG} z} - i \gamma_{SFG} \delta(z) A_2^* A_3 e^{-i \Delta k_{SFG} z} \\
\frac{dA_2}{dz} &= -i2 \gamma_{SFG} \delta(z) A_1^* A_3 e^{-i \Delta k_{SFG} z} - i \gamma_{SHG} \delta(z) A_1^2 e^{i \Delta k_{SHG} z} \\
\frac{dA_3}{dz} &= -i3 \gamma_{SFG} \delta(z) A_1 A_2 e^{i \Delta k_{SFG} z}
\end{aligned}
\quad(1)
$$

* $A_j$ と $k_j$ ($j=1, 2, 3$): それぞれ周波数 $\omega$, $2\omega$, $3\omega$ における電場振幅と波数ベクトル。
* $\Delta k_{SHG}$: SHG（第二高調波発生）の波数ベクトルミスマッチ。
    $$
    \Delta k_{SHG} = k_2 - 2k_1
    $$
* $\Delta k_{SFG}$: SFG（和周波発生）の波数ベクトルミスマッチ。
    $$
    \Delta k_{SFG} = k_3 - k_2 - k_1
    $$
* $\gamma_{SHG}$ と $\gamma_{SFG}$: 対応する非線形強度。
    $$
    \gamma_{SHG} \approx \gamma_{SFG} \approx 4\pi^2 d_{eff} / (n(3\omega)\lambda) \approx 4\pi^2 d_{eff} / (n(2\omega)\lambda) \approx 4\pi^2 d_{eff} / (n(\omega)\lambda)
    $$
* $d_{eff}$: 実効非線形係数（例: ニオブ酸リチウム($\text{LiNbO}_3$)における $d_{eff} = \chi^{(2)}/2 = d_{33}$）。
* $\delta(z)$: 伝播距離 $z$ に沿って、$d_1$ または $d_2$ いずれかのサイズのランダムなドメインを持つ符号が変動するユニティ関数。
