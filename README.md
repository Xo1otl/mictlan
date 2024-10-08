# mictlan

個人的なコードを書くモノレポ

[lean のワークスペースについて](https://github.com/leanprover/lean4/blob/master/src/lake/README.md)
なんか subpackage に manifest がないとか言われるが動いてるので問題なし

# Idea

- 今回考えた state machine を使った設計で、しかのこダイアグラムをポチポチ進められるページを作りたい
- camunda みたいなサービスを go で作りたい
- ターミナルから AI 呼び出せる機能を作る、一つのターミナルにつき一つの会話
  - AI は API Key ではなく Playwright を使って行い無限に話せるようにする
- ねずっちの謎掛けが AI で普通にできそうだからそれを行うサイト作ってみたい
- julia で workspace やりたい時、direnv に julia の alias 書いて workspace ファイルに追加した julia のパッケージを全部 activate すればいい説

# TODO

1. 事前イベントと事後状態で状態遷移図を書く
2. 内部的に、事前イベントと事後状態の間に、中間イベントと事後イベントと取り消しイベントを挿入した状態遷移図に変換する
3. hook を用意して、事前イベントと処理を登録すると、処理を行って事後状態になるか、エラーを返して処理前の状態に戻る仕様を以下のように実装
   1. 事前イベントを発火して中間状態への遷移を試みる、無理だったら処理しないでエラーを返す
   2. 処理を実行する、処理に失敗したら取り消しイベントで事前イベント発火前の状態に戻る。エラーの定義は処理の中で行える
   3. 事後イベントを発火して事後状態へ遷移する、中間状態と事後イベントは自動生成されるので未定義のミスは発生せずかならず遷移できる

- https://www.elastic.co/jp/blog/getting-started-with-the-elastic-stack-and-docker-compose
- awscli と rpk の lazy install 書く

# Memo

- lean のダウンロードはエディタの通知に従ってやる
- 基本 framework,driver 層は自分で書かないけど ui は自作する唯一の infra な気がする
- php の拡張機能は package 分かれていても namespace を共有する。namespace を必ず pkg/module にすれば大丈夫、artifacts に置いてるやつは直すのめんどいので無視
- Entity で Validation 等を DI したい場合に Builder Pattern するのありだと思った。go の context の書き方もあり
- python でも internal とか pkg とか書きたかったけど、PYTHON パスに追加されてフォルダ名が強制的に import で使用されるので、package 名でフォルダ作ることにした。詳しくは package を参照して確認
- direnv が動かん時やターミナルが fish にならない時があるけど code-workspace ファイルでなんどか書き直すと戻る
- julia では、pkg で add IJulia すると jupyter kernel がインストールでき、vscode で julia のカーネル選択すると jupyter notebook で使える
- secrets がなくても devcontainer のビルドは正常にできるけど、ssh key だけ手動で移動する必要がある
- workspace root で`tar -xvf secrets.tar.gz`してからビルドしたら万事解決する

# Note

- devcontainer を閉じる時毎回 close connection をしないと永久に connection が溜まっていく、リセットしたい時は下のコマンドで接続を貼り直す
  - X 転送の socket が消えずに/tmp/.x11-unix が増殖していくから定期的に消さなあかんのかもしれない
  - 開いてる最中にバグるなどしたらコネクションが遺ってしまうのかもしれない
- docker sock を閉じる`netsh interface portproxy delete v4tov4 listenport=2375 listenaddress=10.8.0.2`
- docker sock の forward `netsh interface portproxy add v4tov4 listenport=2375 listenaddress=10.8.0.2 connectaddress=127.0.0.1 connectport=2375`
- `docker context create workstation --docker "host=tcp://10.8.0.2:2375"`
- docker compose ファイルは devcontainer に入ってから scripts/tplgen.sh を実行して生成する
- docker compose up(select services)は devcontainer から volume マウントができない、mount ある場合ホストマシンから実行すべし
- システムのクリップボードも使えるようにする設定(keybindindgs.json は devcontainer ではなくホストマシンの設定)

```json
// Place your key bindings in this file to override the defaults
[
  {
    "key": "ctrl+c",
    "command": "-vscode-neovim.escape",
    "when": "editorTextFocus && neovim.ctrlKeysNormal.c && neovim.init && !dirtyDiffVisible && !findWidgetVisible && !inReferenceSearchEditor && !markersNavigationVisible && !notebookCellFocused && !notificationCenterVisible && !parameterHintsVisible && !referenceSearchVisible && neovim.mode == 'normal' && editorLangId not in 'neovim.editorLangIdExclusions'"
  },
  {
    "key": "ctrl+c",
    "command": "-vscode-neovim.escape",
    "when": "editorTextFocus && neovim.ctrlKeysInsert.c && neovim.init && neovim.mode != 'normal' && editorLangId not in 'neovim.editorLangIdExclusions'"
  },
  {
    "key": "ctrl+l",
    "command": "workbench.action.quickchat.launchInlineChat"
  }
]
```
