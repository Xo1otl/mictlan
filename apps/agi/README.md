# AGI

# 記事テンプレート

```markdown
# 概要
興味を引く概要、あと画像とかで実際どうなるかを最初に示す

# インフラ
docker-compose.yamlを示したり、aws構成図などを用いてインフラの説明などを行う

# 構成
mermaidとかでドメインとかアダプタの図を書いとく

# ドメイン
プロジェクトの中核となる「ドメイン」は、特定の技術（[TECH_EXAMPLE_1]や[TECH_EXAMPLE_2]など）に依存しない、純粋なビジネスロジックを表現する部分です。

たとえば「[CORE_FUNCTIONALITY]」という機能は、[PREREQUISITES_LIST]ことだけを前提に設計できます。[IMPLEMENTATION_DETAILS]は、この時点では考えなくて良いのです。

このように技術的な実装から切り離すことで、仕様変更に強く、テストが書きやすく、コードの意図が明確になります。以下のコードは、そのドメインレイヤーの実装です。

``go
type UserRepository interface {
    FindByEmail(email string) (User, error)
}
``

# アダプタの実装
前章で示したドメインを、実際の技術を使って実装していきます。ここでは具体的に：

[TECH_IMPLEMENTATION_LIST]
使用しています。

先ほどのドメインレイヤーで定義したインターフェース（[INTERFACE_LIST]）に対して、それぞれの具体的な実装（[CONCRETE_IMPLEMENTATION_LIST]）を提供します。この方式のメリットは、例えば[EXAMPLE_REPLACEMENT]に置き換えたい場合、新しい[NEW_IMPLEMENTATION_CLASS]クラスを作るだけで、他のコードは一切変更する必要がないという点です。

また、これらの実装を[INTEGRATION_POINT]として統合することで、[BENEFIT_DESCRIPTION]。以下が各実装のコードです：

``go
type SqlUserRepository struct {
    db *sql.DB
}

func (r MysqlUserRepository) FindByEmail(email string) (User, error) {   
    return user, nil
}
``

# まとめ
全体のまとめと考察
```
