# コーディングルール
`__init__`書くの禁止、必ずファクトリー関数でプロトコルの型にして返す

Enumは〇〇Kind

Interface にはプロパティを持てない、プロパティにも制約をつけたい場合、factory関数の interface の引数で指定する

名詞のインターフェースの命名で困ったら 〇〇able で実装は 〇〇er (Observable Observer など)

コールバック関数について、事前イベントのための関数は 動詞の現在形 + 名詞 、事後イベントのための関数は 名詞 +動詞の過去形 で命名

## イベントについて
* 時間がかかり、不確定要素の強い処理に対してのみ、事前イベントと事後イベントの両方を発火できる必要がある
* 処理の引数だけが重要な場合、事前イベントのみ発火できればよい
* 処理の結果も重要な場合、事後イベントのみ発火できればよい (引数は事後でもprofilerに渡せる)

### 時間がかかるため両方の事前・事後の両方でイベントを発火するもの
* `function.Function.evaluate`
* `function.MutationEngine.mutate`

### 処理の引数だけが重要で、事前イベントだけ発火するもの
いまのところ特になし

### 処理の結果も重要で、事後イベントだけ発火するもの
* `archipelago.Evolver.on_islands_removed`
* `archipelago.Evolver.on_islands_revived`
* `archipelago.Evolver.on_best_island_improved`
* `archipelago.Evolver.on_best_fn_improved`
* `archipelago.Cluster.on_fn_added`
* `archipelago.Cluster.on_fn_selected`

# TODO
* イベントを工夫するか、なんとかして島ごとの進化の追跡など出来たらいいなと思う
* 非同期島モデルできたらもっと効率良くなるけど、ローカルだとそこまでやってもしょうが無い
* LLMの部分もプロトコルにしたり責務の分離など行って拡張性高められる
* ソースコードのパース部分も同様にプロトコル考えて拡張性高めてもいいかも
* 屈折率分散の同定で、分布書くときにjax使ったらscipyのBFGSができない
    * numpyで計算してみて速度確認する、ワンチいける
    * jaxでしか無理なら要所要所でjnp.ndarray->onp.ndarrayの変換を挟めば行けるかなぁ
    * 時間的にフラットさの評価ぐらいまでしかできる気がしない
* 初期値にラウス式与えて、係数を一つ用意して
    * ラウス式を、その係数Bを使って改良せよ、というお題でできるかやってみる

# Idea
* 現状のスコアパターン完全一致のクラスタリング条件は厳格すぎて細分化されそう (やってみたら意外と被るの多いらしくてちゃんとクラスタリングされてた)
* 各テストに対する合否分布や、プログラムの構造でクラスタリングしてみてもいい気がする
* 分極反転幅構造だけでなく、材料の屈折率の波長依存性もコントロールが可能であり、それの探索でLLM-SRできるかもしれない。どんな波長依存性を持つ材料を使えばいいのかについて探索できそう。

# Memo
* 以下の環境変数でjaxのメモリのプリアロケートを制限しないとPCが固まる
    * XLA_PYTHON_CLIENT_PREALLOCATE=false
* 大した計算量じゃないらしく jax より numpy のほうが普通に evaluate が速い
* 強制はしてないけど profiler と mutation engine はシングルトンを想定
* 関数の generics は paramspec でできる、型の渡し方などは Callable と同じ
* bfgs は jaxopts にあるけどこれはメンテされてないから使いたく無かった、しかし発見した npda の式が bfgs でしか収束しなくてびっくり
* inspect.getsource() 使えばコメントを含む関数のソースコードを取得できる
* 島モデルGAはOrchestratorの中で論理的に実装される。実際に島の数だけマルチプロセスするわけではない。(並列処理の為に物理的にCPUを増やさないのと同じ理屈)
* 目的変数4個以上の場合はNSGA-IIIが良いらしい。ZDT3というのでベンチマーできるらしい。

## 改良版プロンプト

### 改良点
* docstring をテンプレに設定する、元のコードで必死にパースして設定してたのが不要になり、docstring かぶりもなくなって複数バージョンあってもすっきり
* 最近の llm は structured output できるのでそれを利用
* 割とコメントに考え書いてくれるし docstring は内容が被りがちなので older versions から削除

### 例
```
You are a helpful assistant exploring scientific mathematical functions. Complete the Python function by changing one or more structures from previous versions to discover a more physically accurate solution.

"""
Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator system with driving force, given data on position, and velocity.
"""

import numpy as np
import scipy

# Initialize parameters
MAX_NPARAMS = 10
PRAMS_INIT = [1.0] * MAX_NPARAMS


def equation_v0(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    return params[0] * x + params[1] * v + params[2]

def equation_v1(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    acceleration = params[0] * x - params[1] * v - params[2] * x**3
    return acceleration

# Improved version of `equation_v1`.
def equation_v2(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ 
    Mathematical function for acceleration in a damped nonlinear oscillator

    Args:
        x: A numpy array representing observations of current position.
        v: A numpy array representing observations of velocity.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.
    """
    

Implement `equation_v2` by **modifying its calculation logic** for improvement, and store the function in the json field.
```
