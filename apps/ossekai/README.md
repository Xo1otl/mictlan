# ossekaiserver

echo で作る

# Coding Style

- 機能毎に module 作った後、interface adapter の実装はその module の中に作る
- infrastructure layerは自分のディレクトリを作ってそこに書く
  - ライブラリのコードを使用する様々な処理はほとんどinterface adapterでありinfraのコードではない
  - インフラのコードというのは主に、ui, driver, framework等で、もしも自作する機会があるならrootにディレクトリを切って作る

# Note

- sub (Subject): ユーザーの一意識別子

  - JWT 標準クレームの一つで、OIDC にも準拠
  - アプリケーション層全体で一貫して使用
  - データベースのプライマリーキーやユーザー参照に利用
  - 例: User.sub, Post.author_sub
  - 抽象的で汎用的な概念のため、認証システムの変更にも柔軟に対応可能

- アプリケーション層の抽象概念:
  - Principal: 認証されたエンティティ（ユーザーやシステム）を表す
  - Subject: Principal と同様、認証されたエンティティを指す。JWT では sub クレームとして使用
  - Identity: ユーザーやエンティティの識別情報を抽象化した概念
  - Claims: エンティティに関する追加情報や属性のセット

# Memo

- GetUser 関数で JWT の検証やってくれるらしい
- jwt だと少し高速化するらしい
- go run cmd/echoapiserver/main.goで行けるっぽい
- kafkaのtopicはrdbのtableみたいなもん

# TODO

- commandserverとqueryserverを分けて二つのcmdを用意し、CQRSを行う
- commandserverは今作られているような感じ
- queryserverでは複雑な検索の対応を考える
