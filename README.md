# mictlan

個人的なコードを書くモノレポ

[lean のワークスペースについて](https://github.com/leanprover/lean4/blob/master/src/lake/README.md)
なんか subpackage に manifest がないとか言われるが動いてるので問題なし

# Hosted Links in this Repo

* [Q&Aサイト](https://answer.mictlan.site)
* [アキネーター](https://akinator.mictlan.site)
* [ブログ](https://blog.mictlan.site)
* [Fediverse](https://lemmy.mictlan.site)
* [IdP](https://auth.mictlan.site)

# Private Linkes in this Repo

* [Mail Server](https://mail.mictlan.site)
* [VPN Server](https://vpn.mictlan.site)
* [Knowledgebase](http://10.8.0.2:3010)
* [ComfyUI](http://10.8.0.2:8188)
* [Fluxgym](http://10.8.0.2:7860)
* [AnythingLLM](http://10.8.0.2:3001)

# Idea

- shebangは`#!/usr/bin/env python`、shellscript引退
- camunda みたいなサービスを go で作りたい
- ターミナルから AI 呼び出せる機能を作る、一つのターミナルにつき一つの会話
  - Playwright を使ってスクレイピング
- julia で workspace やりたい時、direnv に julia の alias 書いて workspace ファイルに追加した julia のパッケージを全部 activate すればいい説

# TODO

* poetry 引退して [uv](https://github.com/astral-sh/uv) に移行する
* uv に移行したのでできてるか試す secrets は export してあるから build cpu container ビルドして gpu ビルドしてみる

## CloudFlare利用
DNSをCloudFlareにしてCloudFlare Tunnelでhttp/httpsはすべてホストしたい
* vpnサーバーはudpなのでcomposeでFQDNではなくipを指定する必要がある (WebUIはhttps)
* 今のmxレコードだとip割れるので隠匿機能が使えない
  * CloudFlare Email Routingを使用する
  * 送信はgmailのsmtpなのでmxレコードいらない
* zrok無くてもこれからcloudflare tunnelでローカルをすぐ公開できるようになる

# Memo

- lean のダウンロードはエディタの通知に従ってやる
- 基本 framework,driver 層は自分で書かないけど ui は自作する唯一の infra な気がする
- php の拡張機能は package 分かれていても namespace を共有する。namespace を必ず pkg/module にすれば大丈夫、artifacts に置いてるやつは直すのめんどいので無視
- Entity で Validation 等を DI したい場合に Builder Pattern するのありだと思った。go の context の書き方もあり
- python でも internal とか pkg とか書きたかったけど、PYTHON パスに追加されてフォルダ名が強制的に import で使用されるので、package 名でフォルダ作ることにした。詳しくは package を参照して確認
- direnv が動かん時やターミナルが fish にならない時があるけど code-workspace ファイルでなんどか書き直すと戻る
- julia では、pkg で add IJulia すると jupyter kernel がインストールでき、vscode で julia のカーネル選択すると jupyter notebook で使える
- secrets がなくても devcontainer のビルドは正常にできるけど、ssh key だけ手動で移動する必要がある
- [これでいいやん](https://github.com/apache/incubator-answer?tab=readme-ov-file)
- nginxなくてもcloudflareで同じことできる
1. 事前イベントと事後状態で状態遷移図を書く
2. 内部的に、事前イベントと事後状態の間に、中間イベントと事後イベントと取り消しイベントを挿入した状態遷移図に変換する
3. hook を用意して、事前イベントと処理を登録すると、処理を行って事後状態になるか、エラーを返して処理前の状態に戻る仕様を以下のように実装
   1. 事前イベントを発火して中間状態への遷移を試みる、無理だったら処理しないでエラーを返す
   2. 処理を実行する、処理に失敗したら取り消しイベントで事前イベント発火前の状態に戻る。エラーの定義は処理の中で行える
   3. 事後イベントを発火して事後状態へ遷移する、中間状態と事後イベントは自動生成されるので未定義のミスは発生せずかならず遷移できる
- Gitの[subtree](https://github.com/git/git/tree/master/contrib/subtree)を使って他のリポジトリに上げつつ同時にmonorepo管理できる
  - `git remote add <repo-name> <url>`
  - `git subtree push --prefix=<path/to/subtree> <repo-name> main`

# Note

## CUDAのインストール
4.7. Windows Subsystem for Linux
These instructions must be used if you are installing in a WSL environment.
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit
```

- vpn用のfirewall設定
  - vpnネットワークをprivateに追加`Set-NetConnectionProfile -InterfaceAlias "<config>" -NetworkCategory Private`
  - 10.8.0.0/24を許可`New-NetFirewallRule -DisplayName "Allow VPN Traffic" -Direction Inbound -Action Allow -Profile Private -RemoteAddress 10.8.0.0/24`
- devcontainer を閉じる時毎回 close connection をしないと永久に connection が溜まっていく、リセットしたい時は下のコマンドで接続を貼り直す
  - X 転送の socket が消えずに/tmp/.x11-unix が増殖していくから定期的に消さなあかんのかもしれない
  - 開いてる最中にバグるなどしたらコネクションが遺ってしまうのかもしれない
- docker contextを作成 `docker context create --docker host="ssh://Username@10.8.0.2" workstation`
- docker sock を閉じる`netsh interface portproxy delete v4tov4 listenport=2375 listenaddress=10.8.0.2`
- docker sock の forward `netsh interface portproxy add v4tov4 listenport=2375 listenaddress=10.8.0.2 connectaddress=127.0.0.1 connectport=2375`
- `docker context create workstation --docker "host=tcp://10.8.0.2:2375"`
- devcontainerのport forwardでstuckする時
  - vpnのmtuをチェックする、うちは1340
  - wslからではなくhostマシンからのsshが可能かどうかを確かめてみる
  - 不可能な場合defenderの影響だと思われる、もし可能なのにforwardできない場合他のエラーがある
  - defenderによる問題が継続する場合ファイアーウォールの設定考える
- docker compose ファイルは devcontainer に入ってから scripts/tplgen.sh を実行して生成する
- docker compose up(select services)は devcontainer から volume マウントができない、mount ある場合ホストマシンから実行すべし
- システムのクリップボードも使えるようにする設定(keybindindgs.json は devcontainer ではなくホストマシンの設定)

# git submoduleの使い方
- `git submodule update --init --recursive`の後`git submodule foreach 'git checkout main'`でDetached状態からブランチに移動する必要がある、それか.gitmodulesでremoteを設定すべきらしい。
- それか`git submodule set-branch`で初期化で`update --remote`するとDetachedではなく特定のbranchを追跡するようにできる。こっちを採用してる
