# 3月21日

## タイトルやアブストからドメインを明確にしていった
```
これってタイトル的に<タイトルに書かれている対象>について機械学習でなんかしようという内容に見受けられるけど、結局どんな感じのデータからどんな感じのデータを計算したいんだろうか。手法はいったんおいといて、ドメインをはっきりさせたい
```

## LLM-SRに決めた
* interatomic: 原子間距離とかポテンシャルエネルギー面を理解してデータセット作るところからだし、機械学習もGNN等高度なモデルを使ってて時間なさそう
* heterogeneous: 不均一触媒についてなもしらん、単一の材料に対する手法の研究というより、様々な材料への適用の評価や比較である。化学の背景知識揃える必要ある参考文献230件あるデータセット作って様々な評価方法を行う時間はない
* LLM-SR: Symbolic Regressionだけど数理的なアルゴリズム控え目で、LLMに丸投げしてる部分がある、ほぼAgent作りでやりやすそう。ソースコードついてる。動作確認とれた

## LLM-SRに取り組む
* LLM-SRを理解しつつ、ソースコードをリファクタリングすることに決めた
* **jaxにしたらnumpy&Adam的なことができるし一番高速**
* **ollama使う設計にしたい(GPTでやると普通に高い)**
* いつもはインターフェースを書いてGPTに実装してもらってるけど、同じ手法を組み込めないか考える

## LLM-SRのコードと論文解読
* 論文に評価後の図が合ってそれらの出力方法がわからん

## 問題の要件の整理
* 従来との違いやアプローチの新しさの整理(新規性)の説明が必要
* 自分の研究テーマとの関連、適用可能性
    * 興味ある範囲への適用可能性も
* 検証のための実装
    * 実装一辺倒にならないように、説明の文章もちゃんと考えていく
    * どの部分に着目してどのような検証を行ったのか明確にする
        * 局所最適解に陥らないためにisland化が必要だとおもってそこがちゃんとできること確認
        * PySRとの精度比較
        
## Adamどこでつかってんのかしらべた
* 関数の形状だけ学習させて、係数の部分はAdamで同定してるっぽい

## 進化的アルゴリズムの勉強が大事な気がしてる

# 3月22日

## 昨日放置した結果確認
![一日放置した結果](image.png)
gemma:12bで試したところ10e-6まで下がっている。ちゃんと探索がうまくいっていてすごい。

## 自分の研究への適用可能性
課題にかかれていた考察事項を考えてみたところ、この手法が自分の研究テーマに直接的に直接的に有効であることがわかった

実際に自分の研究に使用して結果を提出するのもいいかなと思うけど二週間なのがきつい、あと論文の説明と実装がメインテーマなのでそっちに集中しよう

## コピー元があった
https://github.com/google-deepmind/funsearch

LLM-SRはこれのコピペ実装

原文の方が島モデルの解説がわかりやすい

signatureはテストケースに対するスコアパターンから計算されている、clusterはislandの中で

reactみたいにかくなら、useCluster と useIsland の二つのcontextが必要になる？

## 要点をまとめてみた

* 進化的アルゴリズム: island

* 関数スケルトン変異アルゴリズム: LLM

* 関数スケルトンの誤差評価アルゴリズム: 二乗誤差で定義されたLossをAdamでオプティマイズした結果

* サンプリングアルゴリズム: ボルツマン選択

**注意ポイント: islandモデル割とムズイからちゃんと理解する、係数は評価の中で動的に決定される前提であり、係数を含まない関数スケルトンを同定する**

## 新規性をまとめる

アルゴリズムはほぼFunSearch

関数同定の分野では従来は構文木を用いた同定が主流、例えばPySRとの比較での新規性では、構文木の代わりにprogramを使用し、サブツリー変異の代わりにLLMの考えを使っており、性能の向上も示している。探索空間が構文木の時より大きいことを主張している

FunSearchではアルゴリズムの探索をしている。LLM-SRは数式の探索に着目している。

LLM-SRでは、コンテキストを与えたときに一発で正解に近い数式をくれる可能性を数値化したPPLと呼ばれる指標を使った評価を行い、LLMが初めからもっている知識に依存せずに、新しい関数を考え出すことに成功していることを示している点が特に新しい。

PPLと、モデルを用いた探索の収束の最終的な精度や速度、との関係を明らかにしたらおもしろそう

# 3月23日

なもしてない

# 3月24日

samplerやexperience bufferなど、スコープがよくわからんクラスについても把握できてきた

```
Archipelago (群島)
  └── Island (島)
       ├── Laboratory (研究所) - クラスタ選別と進化戦略を担当
       │    └── ClusterSelector - クラスタ選択ロジック
       │    └── MutationEngine - 変異生成ロジック
       └── Cluster (クラスタ)
            └── Function (関数)
```

フルスクラッチの前に、まず一回PoCを使って公式の発見を行ってみたい

NPDAを発見できるか試してみようと思う、計算を回している

# 3月25日

funsearchとしてフルスクラッチすることにしたので[リポジトリ](https://github.com/Xo1otl/funsearch)を作った

submoduleのあれこれのときにミスってqunasysの結果を吹き飛ばした、スコアは2.8だったのが0.59ぐらいまで下がっていた、式が消えてマジで悲しい

# 3月26日

specsに多少ヒントを追加して探索したところ、いい感じの公式が発見されていた。

```
{"sample_order": 6631, "function": "def equation(width: np.ndarray, wavelength: np.ndarray, params: np.ndarray) -> np.ndarray:\n    \"\"\" Mathematical function for shg efficiency\n\n    Args:\n        width: A numpy array representing periodic domain width\n        wavelength: A numpy array representing wavelength.\n        params: Array of numeric constants or parameters to be optimized\n\n    Return:\n        A numpy array representing shg efficiency as the result of applying the mathematical function to the inputs.\n        \"\"\"\n    # Scaling factor for the domain width.  This accounts for variations in the poling period.\n    dw_scaling = params[0] + params[1] / wavelength\n\n    # Calculate the effective domain width.\n    effective_width = width * dw_scaling\n\n    # Total length of the QPM device.  This is determined by the number of domains and the effective width.\n    L = params[2] * effective_width\n\n    # Grating vector.  This is related to the poling period and the wavelength.\n    k_g = params[3]\n\n    # Phase mismatch calculation.  This is the difference between the phase of the fundamental wave and the phase of the second harmonic wave.\n    delta_k = k_g + params[4] - np.pi / effective_width\n\n    # Argument for the sinc^2 function.\n    arg = delta_k * L / 2\n\n    # SHG efficiency calculation. The sinc^2 function is the key element.\n    # The constant factor (params[5]) scales the overall efficiency.\n    # The denominator (arg**2 + params[6]**2) introduces a damping factor that broadens the peak.\n    efficiency = params[5] * L**2 * np.sin(arg)**2 / (arg**2 + params[6]**2)\n\n    # Damping factor to account for losses and imperfections.\n    # This is a polynomial function of the phase mismatch.\n    damping_factor = 2 + params[7] * arg + params[8] * arg**2\n\n    # Apply the damping factor to the efficiency.\n    efficiency = efficiency / damping_factor\n\n    return efficiency", "score": -0.06745296165713882}
```

NPDAを用いない数値解法から得たデータセットを使って、NPDAの公式 (しかもphase_mismatchではなく波長をとしている) が発見された

つまり、一定周期分極反転構造における変換効率の解析的な解放をデータセットから、AI (Gemma3:12b) が自力で発見できることがわかった

結果の確認以外なんもやってない

# 3月27日

* 設計の続きをやろうと思う
    * add_event_listener と generics で置き換えたい
    * evolverはclusterに依存していない。clusterモジュールを作ってそっちにclustered islandみたいにしようかな
    
* おおまかな処理の流れの部分を実装して mock を完成させた、あとは以下を実装する
    * MutationEngine の LLMに新しい関数考えてもらう部分
    * MutationEngine の LLM の出力を 実行可能コードに parse する部分
    * Evaluator をちゃんと作って誤差評価
    * island 内の cluster のボルツマン選択アルゴリズム
    * cluster 内の function の選択アルゴリズム (多分もう完成してる)
    
# 3月28日

* skeleton の形式を変更した、とりあえず実装の一つとしてpython skeletonを作る
    `code_manipulation.py`
    ```python
            self._functions.append(Function(
                name=node.name,
                args=ast.unparse(node.args),
                return_type=ast.unparse(
                    node.returns) if node.returns else None,
                docstring=docstring,
                body='\n'.join(
                    self._codelines[body_start_line:function_end_line]),
            ))
    ```
    
* skeleton 完成してテストもできた、numpy 使いつつ adam オプティマイザで gpu だから爆速
* llm.MutationEngine を作る、まずは mock から
* evaluate を jax にしたら爆速になったが、GPU使い果たしててパソコンから音が鳴るので心配
* MutationEngine でのパースと evaluate が完成した
* island の cluster のボルツマン選択と cluster の function の選択アルゴリズム確認したら終わり

## TODO
* cluster もスコアがないと island でクラスタのボルツマン分布が得られない
* cluster の signature が現在ランダムな文字になってる、ここをスコアの hash などにする
* island の中で cluster のボルツマン選択を行う
* cluster の中で fn を選ぶ処理は、長さを重みとした softmax で選んでて多分完成してる
* 実際のデータで探索を試す、その際ログの出力をより整理する、今はログとりすぎ。あとは普通にミスがないか見直しを行う
* LLM のパースの漏れがまだ結構あるから、バグったやつ情報収集して原因を特定する
* ファクトリ関数を整理したりなどのリファクタリングを行う、論文通りのロジックができているか見直しを行う
* mcp で AI から呼び出せるようにしてみようかなとか考えた

# 3月29日

休憩

# 3月30日

島でのクラスタのボルツマン分布を使った選択と、クラスタでの関数選択アルゴリズム書いて reset で再生成されるのが mock になってたの直した

# 3月31日

docker をアプデしたら起動しなくなり devcontainer を作り直した

データセット一つに対してしか evaluate 出来ない現在の仕様を、３つに対しても evaluate できるように変更する必要がある

- **train** はモデルが学習するデータ、
- **id** は学習データと同様の性質を持ち、モデルの基本的な性能を評価するためのデータ、
- **ood** はモデルの一般化能力やロバスト性を評価するための、未知または異なる分布を持つデータセットです。

3種類に分けて、同時に評価してスコアパターンの完全一致でクラスタリングして、比較ではその平均を使う

まえから思ってたけど完全一致ってほぼ発生しないから、全部くっつけた一個のデータセットでやるのとなんも変わらん気がする

とりあえず自分のコードのテストのために、全部一個にまとめたデータセットで探索試そうと思う、それでも行ける気がしている

その前にもうちょい使いやすいようにリファクタリングを行う。現状では mock 書いてたら完成してたからそのままテストコードでやっている

signature は cluster が持つのではなく function が持つべきな気がしてきたし score と分けることで柔軟な設計ができそう

パッションで adam 使ってみようとしてたけど bfgs のほうが圧倒的に精度よかったので numpy + bfgs のほうが良い結果でるの自明で草

結局3つのデータセットで計算してパターンをシグネチャにするようにできたので、一番カンタンそうなoscillator試す

```
2025-03-31 14:16:39,025 MainThread Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
2025-03-31 14:16:39,026 MainThread Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directo
ry
Using GPU: cuda:0
2025-03-31 14:16:39,453 MainThread Event: on_evaluate
2025-03-31 14:16:44,374 MainThread Event: on_evaluated

(省略)

  -> mutation done with score -7.687119970493464e-06
2025-03-31 14:50:14,477 Thread-2 (_run) Event: on_best_island_improved
    Payload:

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
            acceleration = params[0] + params[1] * v + params[2] * x + params[3] * x * v + params[4] * x**2 + params[5] * x**3
            return acceleration

2025-03-31 15:12:19,131 ThreadPoolExecutor-50_0 Event: on_evaluated
  -> mutation done with score -6.845196973396621e-06
2025-03-31 15:12:19,131 Thread-2 (_run) Event: on_best_island_improved
    Payload:

        def equation_v2(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
            acceleration = params[0] + params[1] * v - params[2] * x + params[3] * x * v - params[4] * x**2 * np.sin(x)
            return acceleration
```

結構すぐいい線いってる式が見つかったっぽい、回数を出力してないから無理やり数えないとわからん、明日はProfilerをちゃんと作る

island のリセットが50分に一回なのにリセット発生してないからあてずっぽう感否めないが、論文でのgpt4oの結果とほぼかわらん精度

id ood train それぞれでスコアが何だったのかも知りたい Profiler で取得できるからそれも記録する

## TODO
* Profiler をちゃんと作る、ステート持てるようにクラスのメソッドにして、スレッドセーフな変数でCountとか計測する、gemini2.5 proにコード丸投げして聞いてみる
* 復習がてらかんたんに docs を作成する
* clustering の基準を改良してみる、合否結果で分けるとか、得意不得意にてるやつ分けるとか、コードの構造で分けるとか、これもgeminiに聞いてみる
* mock いじってたら完成してしまったやつにちゃんと名前つけて、適切なモジュールに配置する
* cluster内で確率NaNが出現するやつに関して原因特定

# 4月1日
* またdockerホストが落ちて止まってた、snapshot取れる機能つけたほうが効率いいかもしれん
* 1000エポック程度だとnpdaは見つからない、oscillator1は500回ぐらいで見つかってた `grep on_evaluated out* | wc -l`

# 4月2日

## TODO
* なんか oscillar1 で evaluate の結果がLLM-SRと違うっぽいので確かめる

リファクタリングを行った、なんも進んで無いけど今後やりやすくはなったはず

# 4月3日
LLM-SRのほうではtrain.csvしか読み込んでなかった、bactgrowでも比較したけどevaluateの収束値はほぼ一緒

docker ホストが落ちるのがメモリリークが原因とわかった ollama と jax 両方がリークしてる

[ollama のメモリリーク](https://github.com/ollama/ollama/issues/10040)は issue が立ってて自分と同じ docker ollama で発生してるらしいしかも gemma3 だけ

それっぽい profiler を作った

## TODO
* best island の更新時に score を表示
* ollama アプデ

# 4月4日
* bactgrowもe-6オーダーの関数が見つかるものだと勘違いをしておりずっと改良を考えていたが、論文見直したら普通に探索に成功していたことに気づいた

ollama アプデして `gemma-3-12b-it-qat-q4_0-gguf` 使ってみたけどメモリリークは治らなかった llama.cpp とか試す時間ないのでコードのミス多いけど qwen2.5-coder 使おうかなと思う

## TODO
* 論文の table 1 と結果を比較する
* PPL の評価はやっぱり大事な気がする、ここが理論的に一番面白いしわかってないとダメな気がする (わかった気がする)
* bfgs の evolver 使わなあかんメンテされてないから使ってなかったけど結局 jaxopts が必須
* てかもうjax使う理由特にないから numpy でいい気がする、けどやっぱ jaxopts にしよう

いろいろ実行してみてバグ探したり検証したりした

自作の方でnpda探索したら普通に0.2ぐらいまで誤差減っててえぐい、多分また探索に成功してる気がする

qwen2.5-coderはNPDAもセルマイヤーの分散式も、そもそもSHGも知らないので全く知識が無くても発見できる可能性を示唆しているかもしれない

LLM-SRで発見した非線形結合モード方程式が、bfgsなら収束するけどlbfgsだと収束できない超特殊な例になっていることに気づいた、bfsg必須かもしれない

poetryをuvに変えたらpip install爆速になった

# 4月5日

npdaによる近似式の再探索頑張ろうと思う

LMXという論文も内容結構にてることがわかった、交差にLLM使うらしい

もう時間が残されてないけど以下の点を質問したい

* 改良について
    * パッと思いつく改良は大体やって、他に思いつくのは評価が大変な改良案が多い (clustering条件、多様性のためにモデルを切り替えながらやってみる等)
    * スコアの下がりかたの図示とか、島ごとのプロファイリングとか詳しい評価がしたいなら改良点はあるけど、重要なのかどうかわからん
* 論文についての検証のための実装で、一番基本的な式の発見は実証済みだけど、以下はまだできてない
    * スコアの下がり方の図示など (ログ見ればできる)
    * 物理的情報なしでの性能、島1個での性能、オプティマイザなし (paramsまで決めてもらう) の性能、プログラムではなくsingle line math formulaを生成してやるときの性能
    * モデルごとのPPLの計算、ノイズを追加したデータでの性能評価
    * 従来の機械学習なしSR手法との精度比較
    * 発見された関数の意味を考える (NPDAの式は結構不思議)
    * FunsearchやLMXとの比較
* 自分の研究テーマに特化させたい、誤差評価の方法を変えてより正確なNPDAの式の探索など、それか周期分極の状態で$\beta$の波長依存性探索とか幅関数探索とかやりたいと考えててそっち方面の改良をしてもよいか

* 一定周期分極反転構造を用いたSHGデバイスの設計において、スペクトル分布が広帯域となるような材料の屈折率分散を探索することを考え、関数同定で検証しようと思った
* 予備実験として、NPDAを用いた非線形結合モード方程式の解析解法の探索を行うことを考えている

自作したほう603回でセルマイヤーいりNPDA解析解法603回で発見してきた、精度くっそ高い、神かもしれん

2時間半回してパースエラー3回だけ、それもカッコの閉じ忘れとかイカれたインデントとか直しようがないものだけ

あまりにもドンピシャな公式を発見してきたから、PPL調べたくなったけど別に直接的に改良にはならないからどうしよう、事前に聞いた感じマジで何も知らなそうだから測るまでも無い気がする

事前知識もあってバグコードを全く生成しないGemmaよりも、何の知識もなくイデントやカッコやMAX_NPARAMSをたまに無視してくるqwenのほうが探索効率が良さげなのは興味深い、ちょっとやんちゃなほうがいい結果発見できる可能性が高い

## TODO

* 自分の研究に沿った改良すすめるのが一石二鳥だし趣旨にも沿ってていいと思う
* 発見した式の定性的な評価はどっかでやっときたいけど、先にデータセット作成用のプログラム書こうかなと思う
* 調べた感じ式の形は違うのに同じ結果が出せてるのは、セルマイヤーの分散式をさらに近似した謎の式発見してる気がするから要検証
* neでもnoでも同様にして近似出来ててすごい、多分理論的に考えてほぼ同じ式出してるんだと思う要検証

# 4月6日

文章いろいろ書いてる

どれやるのが一番いいかわからんけど、geminiと相談して1をやることに決めた

1. フラットさの指標
    * 標準偏差、最大最小の差とかで測れそう？
2. 計算コスト
    * スペクトル分布書くために並列化必須
    * jax.clear_cachesで多分メモリリークは防げてる
    * ollamaの邪魔しないようにメモリ割り当てでバグらんようにする
3. 屈折率分散の設計自由度
    * 量子井戸インターミキシングというキーワードつかんだだけで偉いらしい
    * 仮説ベースですすめる、物理的な実現可能性の検証は課題のスコープ外と割り切る、調べる時間もたらん
    * 物理的に実現不可能でも、物理的な洞察を試みること自体に価値がある
    
そもそもグラフの矩形っぽさどうやって測ればいいかわからん

```
**アプローチのアイデア**

1.  **充填率 (Fullness Ratio):**
    *   グラフの曲線下の面積を、そのグラフを囲む最小の矩形（バウンディングボックス）の面積で割る。
    *   矩形なら1に近くなり、尖った形（例: sinc^2, ガウス関数）では小さくなります。台形も1に近い値になりますが、矩形よりは少し小さくなります。
    *   ベースラインが0でない場合は、`y`の最小値を引いてから計算します。

2.  **トップ部分の平坦度 (Top Flatness):**
    *   グラフの最大値に近い部分（例: 最大値の80%以上）のy値の標準偏差や分散を計算する。
    *   値が小さいほど、トップが平坦であることを意味します。矩形や台形は非常に小さい値になります。
    *   スコアとして使う場合は、この値を逆数にするか、`(1 - 正規化された標準偏差)`のような形にします。

3.  **エッジの急峻さ (Edge Steepness):**
    *   グラフの勾配（微分）を計算します。
    *   矩形は、エッジ部分で非常に大きな勾配を持ち、他の部分ではほぼ0になります。
    *   台形は、傾斜部分で一定の勾配を持ち、他の部分ではほぼ0になります。
    *   尖った関数は、勾配が連続的に変化します。
    *   勾配の最大値や、勾配の分散などを評価できます。矩形らしさを測るなら最大勾配の大きさが指標になりえます。
```

範囲広すぎてもあかん気がする、どうやって定量的に測ればいいのかわからん

対数誤差とかアイデアとしてある

とりあえずなんか進めようと思って残ってる公式を再発見した

# 4月7日

ちょっとずつ結果を出しながら進めたい、異なる周期分極反転構造を5つぐらいくっつけた構造を考えて、それらの長さを関数で表した時に、一番フラットになる形を探索することから始めたい。これぐらいの条件ならば、複雑すぎない関数でフラット(矩形)っぽいものが見つかりそうな予想は立っている

発表までに成果が得られる気がしないけど、頑張ってみる

どうせオプティマイザで同定するので、矩形の範囲まで全部決めて探索しようと思う、OODに対して精度がいいならいけるやろ

フラットさは、規格化してから矩形をかぶせて充填率で評価

* 条件を考える
    * 構造は周期分極反転構造を5つ繋げたものかなぁ
    * 全体の波長幅
    * 矩形になってほしい波長幅
    * ドメイン数or分極反転幅の関数形
    * 温度と基本波初期パワーとSH波初期パワーと$\kappa$
    
ドメイン幅と波長依存性がわかった上で、それをContextとして変数を追加してもっと複雑なやつ試すとどうなるんだろう？

波長範囲が狭すぎてセルマイヤーの式のNがほぼ一定になっている可能性があることに気づいた。

この時変換効率は単純に$\lambda$に反比例するだけなので、そこまで複雑じゃないからすぐ探索できた説が高い

一応研究室的には、見つかったらラッキー見つからなかったら残念、とかではなくなんかステップが踏める進捗があると嬉しいらしい

データに対して〜とあるけど、データ作れない場合は対応せんの？って思う

* 5種類の一定周期分極反転構造でためす
    * 各ブロックの長さを変数として、ドメイン数は固定して、幅はそっから計算
    * duty比をコントロールすれば効率が操作できる
    
# 4月8日

時間ギリギリだけど、急いでファインチューニングやってみたい

geminiでやろうと思って、プログラム対応して探索試したら特にチューニングするところも思い浮かばず、公式が速攻発見できることがわかった

それより複雑な公式にたいしてオプティマイザが目的関数最小化できないのが致命的すぎる

先にオプティマイザで係数決定できるかを調べた上でできるとわかってる関数でほかの発見を試そうかなぁともちょっと思ったけど、geminiはSHGのこと普通にしってそうだ

ていうかfunsearchを競技プログラミングで使ったらどうなるのかがかなり気になる

それなりに文法ミスがあったけど、そもそもの括弧の間違いとか、ファインチューニングで治るもんでもなさそうだからやるならどこをチューニングしたら嬉しいか考えないといけない

設計の説明文を書いた

摂動項を上手に加えるとオプティマイズできる現象について考察したい気もするけど難しすぎる

# 4月9日

最小記述長のことをMDL原理っていうらしい、短い関数表現が優先される理由はこれ

# 4月12日

snapshot機能まじでほしい

ガッツリイベントドリブンで考えて、イベントストアからリードモデル構築する感じにしたい

# 4月15日

改良案としてイベントプロファイラをさらに拡張してproducerとして完全なイベントドリブンにしようかなとか考え出して沼っている

モジュールの分離方法についても改良版ドメイン駆動証明に従ってもっとリファクタリングしたい気持ちがある

evaluator作らなあかん、gemini使うからGPUは空くとして、その場で分布計算をして矩形ぽさとトップの振れ幅の定量評価できるevaluatorがいる

* 分布書く時の条件考えなあかん、フラットにしたい波長範囲とkappaと温度
* デバイス形状振る時の条件考えなあかん、関数形か配列形かどっちもあり得る
    * 5つの周期分極反転構造を並べる
    * それぞれの、ブロックの全長が変数でドメイン数は定数
    * それぞれのduty比も変数

とりまその構造で、スペクトル分布計算するプログラム書いてどんな感じになるか見てみようかなと思う

んでも、関数形の方が個人的にうれしいしワンちood見つかる気もするからできるようにしたい、両方できるプログラム書きたいかなぁ

オプティマイザ困ってるんだった、どうしようもないから摂動項ごと見つけること祈ってやるだけやってみよう

eventソーシングやりたい

eventソーシングにおけるeventログは、コミットされたシステムの状態変化を表すeventのみを残すべきで、アプリケーションログとは別者

今のプロファイラはログのためのプロトコルとしてそのまま残し、eventソーシングのためのプロトコルを用意するんかなぁ

eventソーシングのためのプロトコルでは、各関数の呼び出しで戻り値としてeventを返し、apply_eventsメソッドを定義する感じかなぁ

てかgoのcontext.Contextみたいな仕組みほしいかも、あんま気にしてなかったけど、普通に今stop呼び出してもすべてのスレッドがちゃんと止まってくれない問題はある

やっぱ普通にasyncioやるのがよさげかもしれん、contextvarは使うかどうか迷う
