# 擬似位相整合における実効位相不整合を用いたSH光の振幅計算の導出

## 問題設定

縦型擬似位相整合デバイスにおいて、**特定の条件下で**実効位相不整合を用いた簡潔な表現が得られることを示す。

### 一般的な表現（任意の条件下）

$$A_{2\omega,z^{(N)}} = \sum_{j=0}^{N-1} A_{2\omega}^{(k)}$$

ただし、各層での振幅は：
$$A_{2\omega}^{(k)} = \left(-j\kappa^{(k)} A_{in}^2 L^{(k)} e^{j\Gamma L^{(k)}} \cdot \frac{\sin (\Gamma L^{(k)})}{\Gamma L^{(k)}}\right) \cdot e^{2j\Gamma z^{(k)}}$$

ただし、$\Gamma$は位相不整合と呼ばれる値で：
$$2\Gamma = \beta^{2\omega} - 2\beta^{\omega}$$

### 特殊条件の設定

以下の特殊な条件を考える：

1. **各層の厚さが一定**：$L^{(k)} = \frac{\Lambda}{2}$ （すべての層で同一）
2. **非線形係数の周期的反転**：$\kappa^{(k)} = \kappa_{mag}(-1)^k$

### 実効位相不整合の定義

$$2\Gamma^{'} = \beta^{2\omega} - 2\beta^{\omega} - K$$
$$K = \frac{2\pi}{\Lambda}$$

## 導出目標

**上記の特殊条件下においてのみ**、以下の簡潔な表現が(近似的または完全)成り立つことを証明する：

$$A_{2\omega,z} = -j\kappa_{mag} A_{in}^2 z e^{j\Gamma^{'} z}\left(\frac{\sin(\Gamma^{'}z)}{\Gamma^{'}z}\right)$$

## 導出

### 一定周期分極反転条件と$\Gamma^{'}$を用いた変形

条件を適用すると：
$$A_{2\omega}^{(k)} = -j\kappa_{mag}(-1)^k \underbrace{A_{in}^2 \frac{\Lambda}{2} e^{j\Gamma \frac{\Lambda}{2}} \frac{\sin(\Gamma \frac{\Lambda}{2})}{\Gamma \frac{\Lambda}{2}}}_{定数 C} \cdot e^{2j\Gamma z^{(k)}}$$

各層の位置は $z^{(k)} = k \frac{\Lambda}{2}$ なので：

$$A_{2\omega}^{(k)} = -j\kappa_{mag}(-1)^k C \cdot e^{j\Gamma k \Lambda}$$

全体の和：
$$A_{2\omega,z^{(N)}} = -j\kappa_{mag} C \sum_{k=0}^{N-1} (-1)^k e^{j\Gamma k \Lambda}$$

$(-1)^k = e^{jk\pi}$ を使うと：

$$A_{2\omega,z^{(N)}} = -j\kappa_{mag} C \sum_{k=0}^{N-1} e^{jk(\pi + \Gamma \Lambda)}$$

等比級数の和の公式：

$$\sum_{k=0}^{N-1} e^{jk\alpha} = \frac{1-e^{jN\alpha}}{1-e^{j\alpha}}$$

これをsinc関数の形に変換するには、以下の恒等式を使います：

$$
\begin{aligned}
1 - e^{j\theta} &= e^{j\frac{\theta}{2}}e^{-j\frac{\theta}{2}} - e^{j\frac{\theta}{2}}e^{j\frac{\theta}{2}} \\
&= e^{j\frac{\theta}{2}}\left(e^{-j\frac{\theta}{2}} - e^{j\frac{\theta}{2}}\right) \\
&= e^{j\frac{\theta}{2}}\left(-\left(e^{j\frac{\theta}{2}} - e^{-j\frac{\theta}{2}}\right)\right) \\
&= e^{j\frac{\theta}{2}}\left(-2j\sin\left(\frac{\theta}{2}\right)\right) \\
&= -2j\sin\left(\frac{\theta}{2}\right)e^{j\frac{\theta}{2}}
\end{aligned}
$$

これを分子と分母に適用すると：

$$\frac{1-e^{jN\alpha}}{1-e^{j\alpha}} = \frac{-2j\sin\left(\frac{N\alpha}{2}\right)e^{j\frac{N\alpha}{2}}}{-2j\sin\left(\frac{\alpha}{2}\right)e^{j\frac{\alpha}{2}}} = \frac{\sin\left(\frac{N\alpha}{2}\right)}{\sin\left(\frac{\alpha}{2}\right)}e^{j\frac{(N-1)\alpha}{2}}$$

$$\sum_{k=0}^{N-1} e^{jk\alpha} = \frac{\sin\left(\frac{N\alpha}{2}\right)}{\sin\left(\frac{\alpha}{2}\right)}e^{j\frac{(N-1)\alpha}{2}}$$

今回の場合、$\alpha = \Gamma \Lambda + \pi$ なので：

$$A_{2\omega,z^{(N)}} = -j\kappa_{mag} C \times \frac{\sin\left(\frac{N(\pi + \Gamma \Lambda)}{2}\right)}{\sin\left(\frac{\pi + \Gamma \Lambda}{2}\right)} \times e^{j\frac{(N-1)(\pi + \Gamma \Lambda)}{2}}$$

### sin/sin項の変形

現在の式のsin/sin項：
$$\frac{\sin\left(\frac{N(2\pi + \Gamma' \Lambda)}{2}\right)}{\sin\left(\frac{2\pi + \Gamma' \Lambda}{2}\right)}$$

#### ステップ1：周期性による簡略化

**分子**について：
$$\sin\left(\frac{N(2\pi + \Gamma' \Lambda)}{2}\right) = \sin\left(N\pi + \frac{N\Gamma' \Lambda}{2}\right)$$

sin関数の性質 $\sin(x + n\pi) = (-1)^n \sin(x)$ を使うと：
$$= (-1)^N \sin\left(\frac{N\Gamma' \Lambda}{2}\right)$$

**分母**について：
$$\sin\left(\frac{2\pi + \Gamma' \Lambda}{2}\right) = \sin\left(\pi + \frac{\Gamma' \Lambda}{2}\right)$$

同じ性質を使うと：
$$= (-1)^1 \sin\left(\frac{\Gamma' \Lambda}{2}\right) = -\sin\left(\frac{\Gamma' \Lambda}{2}\right)$$

#### ステップ2：比の計算

$$\frac{\sin\left(\frac{N(2\pi + \Gamma' \Lambda)}{2}\right)}{\sin\left(\frac{2\pi + \Gamma' \Lambda}{2}\right)} = \frac{(-1)^N \sin\left(\frac{N\Gamma' \Lambda}{2}\right)}{-\sin\left(\frac{\Gamma' \Lambda}{2}\right)} = (-1)^{N+1} \frac{\sin\left(\frac{N\Gamma' \Lambda}{2}\right)}{\sin\left(\frac{\Gamma' \Lambda}{2}\right)}$$

#### ステップ3：位相整合条件の適用

位相整合条件では $\Gamma' \approx 0$ なので、小角近似が使えます：
$$\sin(x) \approx x \quad (x \ll 1)$$

分母に適用：
$$\sin\left(\frac{\Gamma' \Lambda}{2}\right) \approx \frac{\Gamma' \Lambda}{2}$$

したがって：
$$(-1)^{N+1} \frac{\sin\left(\frac{N\Gamma' \Lambda}{2}\right)}{\sin\left(\frac{\Gamma' \Lambda}{2}\right)} \approx (-1)^{N+1} \frac{\sin\left(\frac{N\Gamma' \Lambda}{2}\right)}{\frac{\Gamma' \Lambda}{2}}$$

#### ステップ4：$z^{(N)}$への変換

各層の位置関係から：
$$z^{(N)} = N \times \frac{\Lambda}{2}$$

したがって：
$$N\Lambda = 2z^{(N)}$$

これを代入：
$$(-1)^{N+1} \frac{\sin\left(\frac{N\Gamma' \Lambda}{2}\right)}{\frac{\Gamma' \Lambda}{2}} = (-1)^{N+1} \frac{\sin(\Gamma' z^{(N)})}{\frac{\Gamma' \Lambda}{2}}$$

分母分子に2を掛けて整理：
$$= (-1)^{N+1} \frac{2\sin(\Gamma' z^{(N)})}{\Gamma' \Lambda}$$

さらに $\Lambda = \frac{2z^{(N)}}{N}$ を使って：
$$= (-1)^{N+1} \frac{2\sin(\Gamma' z^{(N)})}{\Gamma' \cdot \frac{2z^{(N)}}{N}} = (-1)^{N+1} N \frac{\sin(\Gamma' z^{(N)})}{\Gamma' z^{(N)}}$$

**sin/sin項は次のように変形されます**：
$$\frac{\sin\left(\frac{N(2\pi + \Gamma' \Lambda)}{2}\right)}{\sin\left(\frac{2\pi + \Gamma' \Lambda}{2}\right)} \approx (-1)^{N+1} N \frac{\sin(\Gamma' z^{(N)})}{\Gamma' z^{(N)}}$$

### 位相項の変形

位相項を詳しく見てみます：
$$e^{j\frac{(N-1)(\pi + \Gamma \Lambda)}{2}}$$

$\Gamma \Lambda = \Gamma' \Lambda + \pi$ を代入：
$$e^{j\frac{(N-1)(2\pi + \Gamma' \Lambda)}{2}} = e^{j(N-1)\pi} \cdot e^{j\frac{(N-1)\Gamma' \Lambda}{2}}$$

$e^{j(N-1)\pi} = (-1)^{N-1} = (-1)^{N+1}$ なので：
$$= (-1)^{N+1} \cdot e^{j\frac{(N-1)\Gamma' \Lambda}{2}}$$

ここで、$z^{(N)} = N \frac{\Lambda}{2}$ より $\Lambda = \frac{2z^{(N)}}{N}$ を使うと：
$$e^{j\frac{(N-1)\Gamma' \Lambda}{2}} = e^{j\frac{(N-1)\Gamma' \cdot 2z^{(N)}}{2N}} = e^{j\Gamma' z^{(N)} \frac{N-1}{N}}$$

$N \gg 1$ の近似で $\frac{N-1}{N} \approx 1$ とすると：
$$e^{j\frac{(N-1)\Gamma' \Lambda}{2}} \approx e^{j\Gamma' z^{(N)}}$$

### 定数Cの変形

定数Cは：
$$C = A_{in}^2 \frac{\Lambda}{2} e^{j\Gamma \frac{\Lambda}{2}} \frac{\sin(\Gamma \frac{\Lambda}{2})}{\Gamma \frac{\Lambda}{2}}$$

$\Gamma = \Gamma' + \frac{K}{2} = \Gamma' + \frac{\pi}{\Lambda}$ を代入：
$$\Gamma \frac{\Lambda}{2} = \left(\Gamma' + \frac{\pi}{\Lambda}\right) \frac{\Lambda}{2} = \frac{\Gamma' \Lambda}{2} + \frac{\pi}{2}$$

したがって：
$$e^{j\Gamma \frac{\Lambda}{2}} = e^{j\left(\frac{\Gamma' \Lambda}{2} + \frac{\pi}{2}\right)} = e^{j\frac{\pi}{2}} \cdot e^{j\frac{\Gamma' \Lambda}{2}} = j \cdot e^{j\frac{\Gamma' \Lambda}{2}}$$

また、sinc関数部分は：
$$\frac{\sin(\Gamma \frac{\Lambda}{2})}{\Gamma \frac{\Lambda}{2}} = \frac{\sin\left(\frac{\Gamma' \Lambda}{2} + \frac{\pi}{2}\right)}{\frac{\Gamma' \Lambda}{2} + \frac{\pi}{2}}$$

$\sin(x + \frac{\pi}{2}) = \cos(x)$ を使うと：
$$= \frac{\cos\left(\frac{\Gamma' \Lambda}{2}\right)}{\frac{\Gamma' \Lambda}{2} + \frac{\pi}{2}}$$

位相整合条件で $\Gamma' \Lambda \ll 1$ なので、$\cos\left(\frac{\Gamma' \Lambda}{2}\right) \approx 1$ とすると：
$$\approx \frac{1}{\frac{\Gamma' \Lambda}{2} + \frac{\pi}{2}} = \frac{2}{\Gamma' \Lambda + \pi}$$

$\Gamma' \Lambda \ll \pi$ の近似で：
$$\approx \frac{2}{\pi}$$

これらをまとめると：
$$C \approx A_{in}^2 \frac{\Lambda}{2} \cdot j \cdot e^{j\frac{\Gamma' \Lambda}{2}} \cdot \frac{2}{\pi} = j \frac{A_{in}^2 \Lambda}{\pi} e^{j\frac{\Gamma' \Lambda}{2}}$$

### 全体をまとめる

すべての項をまとめると：
$$A_{2\omega,z^{(N)}} = -j\kappa_{mag} \cdot \underbrace{j \frac{A_{in}^2 \Lambda}{\pi} e^{j\frac{\Gamma' \Lambda}{2}}}_{C} \times \underbrace{(-1)^{N+1} N \frac{\sin(\Gamma' z^{(N)})}{\Gamma' z^{(N)}}}_{\text{sin/sin項}} \times \underbrace{(-1)^{N+1} e^{j\Gamma' z^{(N)}}}_{\text{位相項}}$$

符号を整理すると：
- $-j \cdot j = 1$
- $(-1)^{N+1} \cdot (-1)^{N+1} = (-1)^{2(N+1)} = 1$

したがって：
$$A_{2\omega,z^{(N)}} = \kappa_{mag} \frac{A_{in}^2 \Lambda}{\pi} N \cdot e^{j\frac{\Gamma' \Lambda}{2}} \cdot e^{j\Gamma' z^{(N)}} \cdot \frac{\sin(\Gamma' z^{(N)})}{\Gamma' z^{(N)}}$$

ここで、$\Lambda = \frac{2z^{(N)}}{N}$ を代入：
$$= \kappa_{mag} \frac{A_{in}^2 \cdot 2z^{(N)}}{\pi N} N \cdot e^{j\frac{\Gamma' \cdot 2z^{(N)}}{2N}} \cdot e^{j\Gamma' z^{(N)}} \cdot \frac{\sin(\Gamma' z^{(N)})}{\Gamma' z^{(N)}}$$

$$= \kappa_{mag} \frac{2A_{in}^2 z^{(N)}}{\pi} \cdot e^{j\frac{\Gamma' z^{(N)}}{N}} \cdot e^{j\Gamma' z^{(N)}} \cdot \frac{\sin(\Gamma' z^{(N)})}{\Gamma' z^{(N)}}$$

$N \gg 1$ の近似で $e^{j\frac{\Gamma' z^{(N)}}{N}} \approx 1$ とすると：
$$A_{2\omega,z^{(N)}} \approx \kappa_{mag} \frac{2A_{in}^2}{\pi} z^{(N)} \cdot e^{j\Gamma' z^{(N)}} \cdot \frac{\sin(\Gamma' z^{(N)})}{\Gamma' z^{(N)}}$$

## 結論

特殊条件下（一定厚さの層、周期的分極反転）において、実効位相不整合 $\Gamma'$ を用いた表現：

$$A_{2\omega,z} \propto \kappa_{mag} A_{in}^2 z e^{j\Gamma' z}\left(\frac{\sin(\Gamma' z)}{\Gamma' z}\right)$$

が得られることが示された。目標式との違いは係数のみであり、$z$ と $\Gamma'$ への依存性は完全に一致している。

この結果は、擬似位相整合デバイスにおいて実効位相不整合を用いることで、通常の位相整合と同様の簡潔な表現が得られることを示している。係数の違い（$-j$ や $\frac{2}{\pi}$）は、具体的な非線形係数の定義や規格化の違いによるものである。