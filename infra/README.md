# infra

infrastructure as code

基本pythonで書くこととする

イメージと感覚つかめてきたし、workspace用のpackageとしてツール化したい

## Memo

- サービスによっては複数のドメインで使われる場合がある
- その場合はドメイン固有の設定をドメインのフォルダに書いて、サービス自体のフォルダを別で用意してDIContainerのようにして集約する
- メインのdocker composeではその集約をincludeする
- pythonパッケージとして登録されているのでimportが使える

## 設計

### Poc

これ

### ドメイン

#### インデックス化(ボトムアップで最上位のドメインを決定する)

aws的な感じ、dockerでとりあえずいける、IaCする、pythonで書こうかなと思う、dockerに出てくる概念を踏襲する

docker composeをまとめる、という処理を抽象化したらよさげ

ディレクトリツリーによってカテゴライズされたservicesの木を用意する感じ

それぞれが自身のcontextの中にportやservice_name等の変数を持つ
