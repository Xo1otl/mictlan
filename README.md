# mictlan

monorepoは人生

[lean のワークスペースについて](https://github.com/leanprover/lean4/blob/master/src/lake/README.md)
なんか subpackage に manifest がないとか言われるが動いてるので問題なし

# Idea

- 今回考えた state machine を使った設計で、しかのこダイアグラムをポチポチ進められるページを作りたい
- camunda みたいなサービスを go で作りたい
- ターミナルからAI呼び出せる機能を作る、一つのターミナルにつき一つの会話
  - AIはAPI KeyではなくPlaywrightを使って行い無限に話せるようにする
- ねずっちの謎掛けがAIで普通にできそうだからそれを行うサイト作ってみたい
- juliaでworkspaceやりたい時、direnvにjuliaのalias書いてworkspaceファイルに追加したjuliaのパッケージを全部activateすればいい説

# TODO

1. 事前イベントと事後状態で状態遷移図を書く
2. 内部的に、事前イベントと事後状態の間に、中間イベントと事後イベントと取り消しイベントを挿入した状態遷移図に変換する
3. hook を用意して、事前イベントと処理を登録すると、処理を行って事後状態になるか、エラーを返して処理前の状態に戻る仕様を以下のように実装
   1. 事前イベントを発火して中間状態への遷移を試みる、無理だったら処理しないでエラーを返す
   2. 処理を実行する、処理に失敗したら取り消しイベントで事前イベント発火前の状態に戻る。エラーの定義は処理の中で行える
   3. 事後イベントを発火して事後状態へ遷移する、中間状態と事後イベントは自動生成されるので未定義のミスは発生せずかならず遷移できる
4. poetryどうしようか考える
   
- docker sockのforward `netsh interface portproxy add v4tov4 listenport=2375 listenaddress=10.8.0.2 connectaddress=127.0.0.1 connectport=2375`
- GPUがないと動かないパッケージをどう分けるか考える
- meepがpoetryからインストール不可能だけど、nixでpoetryとmeepをインストールするとどっちも使える
- `docker context create workstation --docker "host=tcp://10.8.0.2:2375"`

# Memo

- leanのダウンロードとcomposerはエディタの通知に従ってやる
- 基本framework,driver層は自分で書かないけどuiは自作する唯一のinfraな気がする
- phpの拡張機能がポンコツすぎてpackage分かれていてもnamespaceを共有するためmulti package workspaceのlintが不可能、エラーでるのどうしようもない
- EntityでValidation等をDIしたい場合にBuilder Patternするのありだと思う
- pythonでもinternalとかpkgとか書きたかったけど、PYTHONパスに追加されてフォルダ名が強制的にimportで使用されるので、package名でフォルダ作ることにした。詳しくはpackageを参照して確認
- direnvが動かん時やターミナルがfishにならない時があるけどcode-workspaceファイルでなんどか書き直すと戻る
- juliaでは、pkgでadd IJuliaするとjupyter kernelがインストールでき、vscodeでjuliaのカーネル選択するとjupyter notebookで使える

# Note

- docker compose up(select services)はdevcontainerからできない、volumeマウントができない、ホストマシンから実行すべし
- Oracle.mysql-shell-for-vs-codeもバリ便利やけどdevcontainerから動かない、new windowしてホストからは見れる
- システムのクリップボードも使えるようにする(keybindindgs.jsonはdevcontainerではなくホストマシンの設定)
- neovimとOracleのmysql shellはlocalで入れるべし
- システムのクリップボードを別で使うための設定
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
