## $s \cdot x + y \cdot x$と$(x \oplus y) \cdot x$の偶奇が常に一致する証明

はい、ご提示の2つの式の偶奇（パリティ、すなわち2で割った余り）が常に一致することを示すことができます。

ここで、記号の定義と演算が行われる体（フィールド）について確認します。
- $s, x, y$ は $n$ ビットのバイナリ列（ベクトル）とします。各要素（ビット）は 0 または 1 です。
- $a \cdot b$ はビットごとの内積（ドット積）を表し、$a \cdot b = \sum_i a_i b_i \pmod 2$ と計算されます。結果は 0 または 1 になります。
- $a \oplus b$ はビットごとの排他的論理和（XOR）を表します。
- 式 $A+B$ の「偶奇」とは、$(A+B) \pmod 2$ を意味します。

ご提示の2つの式は以下のように解釈できます。

1.  **第1式: $s \cdot x + y \cdot x$ の偶奇**
    $s \cdot x$ の結果は 0 または 1 です。同様に $y \cdot x$ の結果も 0 または 1 です。
    これらの和 $s \cdot x + y \cdot x$ は、0 (0+0), 1 (0+1 または 1+0), または 2 (1+1) のいずれかの値を取ります。
    この和の偶奇は、$(s \cdot x + y \cdot x) \pmod 2$ です。
    これは、$(s \cdot x) \oplus (y \cdot x)$ と等価です（例えば、$0+0=0 \equiv 0 \oplus 0$、$1+0=1 \equiv 1 \oplus 0$、$1+1=2 \equiv 0$, そして $1 \oplus 1 = 0$）。
    したがって、第1式は $(s \cdot x) \oplus (y \cdot x)$ と書けます。

2.  **第2式: $(s \oplus y) \cdot x$ の偶奇**
    この式は、まず $s$ と $y$ のビットごとのXORを計算し、その結果と $x$ の内積を取ります。内積の定義により、結果は既に 0 または 1 なので、その偶奇は値そのものです。
    したがって、第2式は $(s \oplus y) \cdot x$ と書けます。

証明すべきことは、以下の等式が成り立つことです。
$$(s \cdot x) \oplus (y \cdot x) = (s \oplus y) \cdot x$$

ここで、全ての演算は $\mathbb{F}_2$（2元体、つまり $\{0,1\}$ 上の体で、加算はXOR、乗算はANDに相当）で行われると考えます。
内積の定義 $a \cdot b = \bigoplus_i (a_i \land b_i)$ （ここで $\land$ は論理AND）を用いると、

左辺は、
$$(\bigoplus_i (s_i \land x_i)) \oplus (\bigoplus_j (y_j \land x_j))$$
XOR演算の結合律と交換律により、これは次のように書き換えられます。
$$\bigoplus_i ((s_i \land x_i) \oplus (y_i \land x_i))$$

右辺は、
$$(s \oplus y) \cdot x = \bigoplus_i ((s_i \oplus y_i) \land x_i)$$

したがって、私たちが示すべきことは、各ビット $i$ について以下の等式が成り立つことです。
$$(s_i \land x_i) \oplus (y_i \land x_i) = (s_i \oplus y_i) \land x_i$$
これは、$\mathbb{F}_2$ における乗算（AND）の加算（XOR）に対する分配法則です。つまり、$ac \oplus bc = (a \oplus b)c$ という形をしています（ここで $a=s_i, b=y_i, c=x_i$）。

この分配法則が成り立つことを真理値表で確認してみましょう（$A=s_i, B=y_i, C=x_i$ とします）。

| $A$ | $B$ | $C$ | $A \land C$ | $B \land C$ | $(A \land C) \oplus (B \land C)$ | $A \oplus B$ | $(A \oplus B) \land C$ |
|-----|-----|-----|-------------|-------------|------------------------------------|--------------|--------------------------|
| 0   | 0   | 0   | 0           | 0           | 0                                  | 0            | 0                        |
| 0   | 0   | 1   | 0           | 0           | 0                                  | 0            | 0                        |
| 0   | 1   | 0   | 0           | 0           | 0                                  | 1            | 0                        |
| 0   | 1   | 1   | 0           | 1           | 1                                  | 1            | 1                        |
| 1   | 0   | 0   | 0           | 0           | 0                                  | 1            | 0                        |
| 1   | 0   | 1   | 1           | 0           | 1                                  | 1            | 1                        |
| 1   | 1   | 0   | 0           | 0           | 0                                  | 0            | 0                        |
| 1   | 1   | 1   | 1           | 1           | 0                                  | 0            | 0                        |

真理値表の6列目「$(A \land C) \oplus (B \land C)$」と8列目「$(A \oplus B) \land C$」がすべての組み合わせで一致しているため、ビットごとの等式 $(s_i \land x_i) \oplus (y_i \land x_i) = (s_i \oplus y_i) \land x_i$ が成り立ちます。

各ビットについてこの関係が成り立つので、それらを全体でXOR総和（$\bigoplus_i$）を取ったものも等しくなります。
$$\bigoplus_i ((s_i \land x_i) \oplus (y_i \land x_i)) = \bigoplus_i ((s_i \oplus y_i) \land x_i)$$
これは、
$$(s \cdot x) \oplus (y \cdot x) = (s \oplus y) \cdot x$$
を意味します。

したがって、$s \cdot x + y \cdot x$ の偶奇と $(s \oplus y) \cdot x$ の偶奇は常に一致します。これは内積操作が（$\mathbb{F}_2$ 上で）線形性を持つことの現れの一つです。

## rangeとpre imageを使った式

$$\frac{1}{2^n} \sum_{x \in \{0,1\}^n} (-1)^{x \cdot y} |f(x)\rangle = \frac{1}{2^n} \sum_{z \in \text{range}(f)} \left[ \sum_{x \in f^{-1}(\{z\})} (-1)^{x \cdot y} \right] |z\rangle$$

Awesome video, cleared up so much about how $f(x)$ influences measurements!

1.  **Phase Kickback:** Really understood how $U_f|->|x> = (-1)^{f(x)}|->|x>$ works! The $|x>$ register gets an 'invisible' phase (its basis state is unchanged by this), which then alters interference after Hadamards, changing what's measured. Hidden phase, big observable impact!

2.  **Simon's Algorithm:** Great explanation of its different approach! It uses $U_f$ for entanglement to create $\frac{1}{\sqrt{2^n}}\sum_x |f(x)>|x>$, so no direct $(-1)^{f(x)}$ kickback onto $|x>$ like above. But $f(x)$'s $s$-periodicity (from the promise $f(x) = f(x \oplus s)$) is crucial. After more Hadamards on the first register, the math beautifully leads to measuring only $y$ values where $y \cdot s = 0 \pmod 2$. So $f(x)$'s structure still powerfully shapes the $y$ measurement outcomes.

3.  **A Thought This Sparked:** Since $U_f$ creates entanglement, it made me wonder: even if the rest of the circuit were built more 'randomly,' would $f(x)$'s influence still remain in the measurement probabilities in *some* form?

Thanks for the great explanations in the video!