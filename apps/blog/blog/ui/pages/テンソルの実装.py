import streamlit as st
import torch
import math

r"""
# テンソルの実装

※演習の回答はtorch.tensorから実際に取得した値です

## 命題: 任意のテンソルの任意のスライスをO(1)で作成できる

入力: テンソルとスライス

出力: スライスされたテンソル

境界条件: 計算時間がO(1)

### 具体例

スライスは、各次元に対してstart:stop:stepを指定するものです

スライスの、ある次元の値を固定した場合、その次元は省略されます

ここで、3x4x5x6のテンソルを、[::2, 1:3, 2, :]でスライスすることを考えます。これを各次元ごとに見ていきましょう。

*   **`::2` (最初の次元)**

    *   `:` は「最初から最後まで」を意味します。
    *   `::2` は「最初から最後まで、2つおきに」という意味になります。
    *   元のテンソルは最初の次元が `3` なので、インデックス `0` から始まり、`2` つおき、つまり `0` と `2` の要素が選ばれます。
*   **`1:3` (2番目の次元)**

    *   `1:3` は「インデックス `1` から、`3` の手前まで（つまり `2` まで）」という意味です。
    *   元のテンソルは2番目の次元が `4` なので、インデックス `1` と `2` の要素が選ばれます。
*   **`2` (3番目の次元)**

    *   `2` は「インデックス `2` の要素だけ」という意味です。
    *   元のテンソルは3番目の次元が `5` なので、インデックス `2` の要素が選ばれます。
*   **`:` (4番目の次元)**

    *   `:` は「最初から最後まで全部」という意味です。
    *   元のテンソルは4番目の次元が `6` なので、インデックス `0` から `5` までのすべての要素が選ばれます。

**4. スライスの結果**

`[::2, 1:3, 2, :]` でスライスした結果、以下の要素が選ばれます。

*   最初の次元: インデックス `0` と `2` の **2つ**
*   2番目の次元: インデックス `1` と `2` の **2つ**
*   3番目の次元: インデックス `2` の **1つ**
*   4番目の次元: インデックス `0` から `5` の **6つ**

これらの要素を組み合わせると、`2 x 2 x 1 x 6` のテンソルになります。ここで、3番目の次元は固定されているので、通常は省略され、最終的に **`2 x 2 x 6` のテンソル** が得られます。

### 処理: 実装方法
* 補題1: 一次元配列とストライドでテンソルを表現できる
* 補題2: 一次元配列がスライスしたテンソルに見えるようなストライドを作成できる

補題1,2より、任意のテンソルを一次元配列とストライドで表現し、ストライドを工夫することで、スライスしたテンソルを表現できます

また、この時の計算時間はO(1)です

### 演習: 入出力と境界条件の確認
"""

tensor_shape_q1 = (3, 4, 5, 6)
slices_q1 = (slice(None, None, 2), slice(None, 4, 2), 2, slice(None))
tensor_q1 = torch.arange(math.prod(tensor_shape_q1)).reshape(tensor_shape_q1)
sliced_tensor_q1 = tensor_q1[slices_q1]

st.write(
    f"* 質問: {tensor_shape_q1}のテンソルを、{slices_q1}でスライスして得られるテンソルの形状は何か、処理にかかる計算時間はO(1)か")

if st.button("回答を表示 (演習1)"):
    st.write(f"* 回答: {sliced_tensor_q1.shape}のテンソルが得られ、計算時間はO(1)である")

r"""
## 補題1: 一次元配列とストライドでテンソルを表現する

入力: テンソル

出力: 一次元配列とストライドを用いたテンソルの表現

### 具体例

3x4x5x6のテンソルは、長さ360の一次元配列と、(120, 30, 6, 1)のストライドになる

**ストライドとは？**

ストライドは、**多次元配列を一次元配列で表現したときに、ある次元のインデックスが1つ増えたときに、一次元配列上で何個分のインデックスが移動するか** を表す値です。

今回の例では、`3x4x5x6` のテンソルを考えています。これをメモリ上に一次元的に配置することを考えます。この時、各次元のインデックスが1増えると、一次元配列上ではどれだけ移動すれば良いのでしょうか？それがストライドです。

*   **最初の次元 (3)**: 最初の次元のインデックスが1増えると、その背後にある `4x5x6 = 120` 個の要素を飛び越える必要があります。つまり、一次元配列上では `120` 個分移動します。
*   **2番目の次元 (4)**: 2番目の次元のインデックスが1増えると、その背後にある `5x6 = 30` 個の要素を飛び越える必要があります。つまり、一次元配列上では `30` 個分移動します。
*   **3番目の次元 (5)**: 3番目の次元のインデックスが1増えると、その背後にある `6` 個の要素を飛び越える必要があります。つまり、一次元配列上では `6` 個分移動します。
*   **4番目の次元 (6)**: 4番目の次元のインデックスが1増えると、飛び越える要素はありません。なぜなら、これが最後の次元だからです。つまり、一次元配列上では `1` 個分移動します。（形式的に1と考えることが多いです）

したがって、ストライドは `(120, 30, 6, 1)` となります。

そして、テンソルにおける座標(i,j,k,l)の値は、一次元配列では $i \times 120 + j \times 30 + k \times 6 + l$ でアクセスできます

### 処理: 実装方法

テンソルのすべての要素を並べて一次元配列を作成します

テンソルのそれぞれの次元で1移動する時、一次元配列上でどれだけ移動すれば同じ値に到達できるか計算します

3x4のテンソルの場合、ストライドは以下のように計算されます

* サイズ4の次元で1移動すると、一次元配列でも1移動するため、サイズ4の次元のストライドは1です
* サイズ4の次元で4移動しようとすると、サイズ3の次元で1移動します
* これは一次元配列で4移動することに対応するため、サイズ3の次元のストライドは4です

したがって、3x4テンソルのストライドは(4, 1)になります

同様にして、3x4x5x6のテンソルのストライドは(120, 30, 6, 1)になります

### 演習: 処理の確認
"""

tensor_shape_q2 = (2, 2, 2)
tensor_q2 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

st.write(
    f"* 質問: {tensor_shape_q2}のテンソル{tensor_q2.tolist()}の一次元配列表記とストライドを計算してください")

if st.button("回答を表示 (演習2)"):
    flat_tensor_q2 = tensor_q2.flatten().tolist()
    strides_q2 = (4, 2, 1)
    st.write(f"* 回答: {flat_tensor_q2}となり、ストライドは{strides_q2}です")

r"""
## 補題2: 一次元配列がスライスしたテンソルに見えるようなストライドが計算できる

入力: テンソルとスライス

出力: ストライドを用いた、スライスしたテンソルの表現

### 具体例

例えば、3x4x5x6のテンソルを [::2, 1:3, 2, :] でスライスする場合ストライドは以下のようになります。

* ストライド: (240, 30, 6, 1)
* オフセット: 42
* サイズ: (2, 2, 6)

### 処理: 実装方法

スライス操作 `tensor[slice_0, slice_1, ..., slice_n]` は、内部的には以下の手順で実現されます。

1. **各次元のスライス情報に基づいて、新しいサイズ、オフセット、ストライドを計算する。**
2. **元のテンソルの情報と、新しいサイズ、オフセット、ストライドを組み合わせて、新しいテンソル（ビュー）を作成する。**
3. **要素アクセス時は、新しいストライドとオフセットを使って、元のテンソルのデータにアクセスする。**

具体例として、3x4x5x6のテンソルを `[::2, 1:3, 2, :]` でスライスする場合を考えます。

* **元のテンソルの情報:**
    * 一次元配列: 3 * 4 * 5 * 6 = 360 個の要素を持つ一次元配列
    * 元のストライド: (120, 30, 6, 1)
    * 元のサイズ: (3, 4, 5, 6)

* **スライス操作の情報:** `[::2, 1:3, 2, :]`
    * `::2`: 最初の次元を、ステップ2でスライス。
    * `1:3`: 2番目の次元を、1から3未満まで（1と2）スライス。
    * `2`: 3番目の次元をインデックス2で固定。スライス後のテンソルでは省略される。
    * `:`: 4番目の次元は全て選択。

* **新しいテンソル(ビュー)の計算:**

    * **オフセット:**
        * スライスされたテンソルの最初の要素は、元のテンソルの(0, 1, 2, 0)の位置にあります。
        * この位置を元のテンソルの一次元配列のインデックスに変換すると、0 * 120 + 1 * 30 + 2 * 6 + 0 * 1 = 42 となります。
        * したがって、オフセットは **42** です。
    * **サイズ:**
        * 最初の次元: `::2` より、ceil((3-0)/2) = 2
        * 2番目の次元: `1:3` より、3 - 1 = 2
        * 3番目の次元: `2` より、1 だが、インデックス指定してるため省略
        * 4番目の次元: `:` より、6
        * 結果のサイズ: **(2, 2, 6)**
    * **ストライド:**
        * 最初の次元: `::2` より、元のストライド 120 にステップ 2 を掛けて、**240**
        * 2番目の次元: `1:3` より、元のストライド 30 にステップ 1 を掛けて、**30**
        * 3番目の次元: 元のストライド 6 は変わらず **6** だが、インデックス指定してるため省略
        * 4番目の次元: `:` より、元のストライド 1 は変わらず **1**
        * 結果のストライド: **(240, 30, 1)**

* **要素へのアクセス:**
    * スライスされたテンソルの要素にアクセスする際は、**指定されたインデックスと新しいストライドを用いて、元のテンソルの一次元配列における相対位置を計算し、オフセットを加算することで絶対位置を求めます。**
    * 例えば、スライスされたテンソルの `(1, 0, 3)` の要素にアクセスしたい場合、
        * サイズ `(2,2,6)` を超えていないことを確認します。
        * `1 * 240 + 0 * 30 + 3 * 1 = 243` を計算します。
        * 3番目の次元やstartの情報はオフセットを計算する際に加味されています。
        * オフセット `42` を加算して、`243 + 42 = 285` が、一次元配列における絶対位置となります。

このように、**新しいストライド、オフセット、サイズ** を使えば、元のテンソルの一次元配列から、スライスされたテンソルの各要素へアクセスすることができます。これにより、新しいメモリ領域を確保することなく、**O(1)** の計算時間でスライス操作を実現しています。

### 演習: 処理の確認
"""

# 演習問題の生成
tensor_shape2 = (2, 3, 4, 5)
slices2 = (slice(None, None, 2), 1, slice(1, 4, 2), slice(None))
tensor2 = torch.arange(math.prod(tensor_shape2)).reshape(tensor_shape2)
flattened_tensor2 = tensor2.flatten()
sliced_tensor2 = tensor2[slices2]

slice3 = (0, 1, 3)
element = sliced_tensor2[slice3]
linear_index = (flattened_tensor2 == element).nonzero(
    as_tuple=True)[0].item()

st.write(
    f"* 質問: {tensor_shape2}のテンソルを {slices2} でスライスするとき、新しいストライド、オフセット、サイズを求めてください。")

if st.button("回答を表示 (演習3)"):
    f"""
    * ストライド: {sliced_tensor2.stride()}
    * オフセット: {sliced_tensor2.storage_offset()}
    * サイズ: {sliced_tensor2.size()}
    """

st.write("* 質問: このスライスされたテンソルの中で、(0, 1, 3) の要素は、元のテンソルの一次元配列の何番目に位置するか、計算してください")

if st.button("回答を表示 (演習4)"):
    st.write(f"* 回答: 一次元配列において、(0, 1, 3) の要素のインデックスは `{linear_index}` です")

r"""
## まとめ

以上により、任意のテンソルを一次元配列とストライドで表現し、任意のスライスに対応するストライドを計算することで、スライスしたテンソルをO(1)で表現できます

pandas.DataFrameのようなデータ構造としても応用でき、多次元データをO(1)で直感的に扱えてプログラミング言語のネイティブ型に組み込んでくれていいぐらい汎用性高いと思います

多次元データの操作をテンソルを使った直積や縮約やマスクといった解釈にすると、多重for loopをテンソル同士の演算に置き換えることができてフレームワークを用いたGPU対応ができます

### 具体例

[[1,2,3], [4,5,6], [7,8,9]]のような行列に対して、行の情報の取得はO(1)でできるが、列情報を取得する時はfor loopを回す必要がある。

テンソルを使えばメモリ使用量も変わらす、両方O(1)になる
"""
