# mictlan

# Setup
## VPNの接続
- vpn用のfirewall設定
  - vpnネットワークをprivateに追加`Set-NetConnectionProfile -InterfaceAlias "<config>" -NetworkCategory Private`
  - 10.8.0.0/24を許可`New-NetFirewallRule -DisplayName "Allow VPN Traffic" -Direction Inbound -Action Allow -Profile Private -RemoteAddress 10.8.0.0/24`
- devcontainer を閉じる時毎回 close connection をしないと永久に connection が溜まっていく、リセットしたい時は下のコマンドで接続を貼り直す
  - X 転送の socket が消えずに/tmp/.x11-unix が増殖していくから定期的に消さなあかんのかもしれない
  - 開いてる最中にバグるなどしたらコネクションが遺ってしまうのかもしれない
## Dockerの接続
- docker contextを作成 `docker context create --docker host="ssh://Username@10.8.0.2" workstation`
- devcontainerのport forwardでstuckする時
  - vpnのmtuをチェックする、うちは1340
  - wslからではなくhostマシンからのsshが可能かどうかを確かめてみる
  - 不可能な場合defenderの影響だと思われる、もし可能なのにforwardできない場合他のエラーがある
  - defenderによる問題が継続する場合ファイアーウォールの設定考える
## Devcontainerの作成
0. wslでcloneし、vscodeで開く
1. submoduleをupdateする
2. secretsを配置する
3. reopen in containerする

# Submoduleについて
- `git submodule update --init --recursive`の後`git submodule foreach 'git checkout main'`でDetached状態からブランチに移動する必要がある、それか.gitmodulesでremoteを設定すべきらしい。
- それか`git submodule set-branch`で初期化で`update --remote`するとDetachedではなく特定のbranchを追跡するようにできる。こっちを採用してる

# Memo
- `bundle .secrets secretm.tar.gz`などでクレデンシャルを全部バンドルして運ぶと便利(個人専用、share禁止)
- shebangは`#!/usr/bin/env python`、shellscript引退
- [lean のワークスペースについて](https://github.com/leanprover/lean4/blob/master/src/lake/README.md)
- lean のダウンロードはエディタの通知に従ってやる
- julia では、pkg で add IJulia すると jupyter kernel がインストールでき、vscode で julia のカーネル選択すると jupyter notebook で使える
- secrets がなくても devcontainer のビルドは正常にできるけど、ssh key だけ手動で移動する必要がある
1. 事前イベントと事後状態で状態遷移図を書く
2. 内部的に、事前イベントと事後状態の間に、中間イベントと事後イベントと取り消しイベントを挿入した状態遷移図に変換する
3. hook を用意して、事前イベントと処理を登録すると、処理を行って事後状態になるか、エラーを返して処理前の状態に戻る仕様を以下のように実装
   1. 事前イベントを発火して中間状態への遷移を試みる、無理だったら処理しないでエラーを返す
   2. 処理を実行する、処理に失敗したら取り消しイベントで事前イベント発火前の状態に戻る。エラーの定義は処理の中で行える
   3. 事後イベントを発火して事後状態へ遷移する、中間状態と事後イベントは自動生成されるので未定義のミスは発生せずかならず遷移できる
