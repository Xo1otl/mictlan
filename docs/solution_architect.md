# 設計手順

## 1. PoC作成
いきなりドメイン作成は難しいのでPoCを書いてイメージを掴みます。

## 2. ドメイン作成

### 1. インデックス化
入力コーパスをTextUnit（テキストユニット）に分割します。これらは、プロセスの残りの部分で分析可能な単位として機能し、出力において詳細な参照を提供します。
LLMを使用して、TextUnitから全ての実体（エンティティ）、要約、関係性、重要な主張を抽出します。
Leiden手法を用いてグラフの階層的クラスタリングを実行します。これを視覚的に確認するには、上記の図1をご覧ください。各円は実体（例：人物、場所、組織）を表し、大きさはその実体の次数を、色はそのコミュニティを表しています。
各コミュニティとその構成要素のサマリーをボトムアップで生成します。これによりデータセットの全体的な理解を助けます。

### 2. Proof Tree作成
最上位のドメインを再帰的にサブドメインに分割します。
構成図では、ドメインの内容を木構造で表します。

```python
from typing import List


class Lemma:
    """補題を表現するクラス。

    各補題は、他の補題（self.lemmas）が証明済みであることを前提として
    証明を構築することができます。

    Attributes:
        statement: 補題の内容を表現する文
        lemmas: この補題の証明で前提として使用する補題のリスト
    """

    def __init__(self, statement=None):
        self.statement = statement
        self.lemmas = []


class SolutionArchitect:
    def __init__(self, context_repository):
        self.context_repository = context_repository

    def build(self, theorem: Lemma) -> Lemma:
        """証明木を構築する。

        与えられた定理を基本補題まで再帰的に分解し、proof treeを構築します。
        構築後は、基本補題から順に証明を行うことで、全体の証明が完成します。

        Args:
            theorem: 証明したい定理・補題

        Returns:
            証明木が構築された定理
        """
        if self.is_primitive(theorem):
            return theorem

        for lemma in self.decompose(theorem):
            theorem.lemmas.append(self.build(lemma))

        return theorem

    def is_primitive(self, lemma: Lemma):
        """
        基本補題（primitive lemma）かどうかを判定
        - 他の補題から導出されない基本的な命題
        - これ以上分解すると本質的な意味が失われる
        - 公理や定義から直接証明できる
        などの条件を想定
        """
        ...

    def decompose(self, lemma: Lemma) -> List[Lemma]:
        """
        補題をより基本的な補題に分解
        - 結合された命題の分離
        - 証明のステップへの分解
        などの分解方法を想定
        """
        ...


# TODO: 前述のGraphRAGを利用したContextの取得が可能なリポジトリが必要
context_repository = ...

theorem = Lemma("Complex theorem statement")
proof_tree = SolutionArchitect(context_repository).build(theorem)
```
proof treeができたら、すべての基本補題をMockを用いて実装し、それらを組み合わせて原理実証を行います。

## 3. インフラ準備
データベースや外部サービスなどのインフラを準備します。
例えば、`docker compose up`したりawsのインスタンス建てたりします。

## 4. アダプタ作成
ドメインで定義したインターフェースに従って、実際のインフラ（データベースや外部サービス）を使用した実装を行います。すべてのMockを置き換えたら完成です。
