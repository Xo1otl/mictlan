# mictlan

monorepo

import 可能な package 同士を言語やランタイム毎にカテゴライズしてみた

python はワークスペースの概念ないけど**init**おけばどこでも package になって相対パスで import できるので mictlan のルートで設定しとけば多分問題ない

jupyter はプロジェクトで使うというよりスクリプトとして使うのでどこかの workspace に所属するというより mictlan のルートから参照を行う

aws sdk go を見習って internal を外部 package にするかも

[lean のワークスペースについて](https://github.com/leanprover/lean4/blob/master/src/lake/README.md)
なんか subpackage に manifest がないとか言われるが動いてるので問題なし

パッケージの規模感がでかいので、hashicorp のリポジトリにありがちな名前を付けたほうが良い気がする

そこで ptrlib のようにニックネームを付けることにする

だからニックネーム考える

# Coding Style

1. 基本構造:

   - `pkg`: 再利用可能なコード
      - みんなlibにしてるけど、再利用可能なコードはinternalかpkgにしか書かない、libのコードはすべて、外部に公開され再利用可能なコード、だからpkgを露出させた。aws sdk go v2のinternal的な感じにしたつもり、あれ確かmodがネストした構造になってたはず
   - `internal`: アプリケーション固有のコード
   - `cmd`: メイン関数
   - `web`: 公開される Web コード

2. モジュール化:

   - 機能ごとにモジュールを分ける（例：`js/pkg/auth`）
   - インフラ層のコードは機能とインフラ名がわかるパッケージ名にする（例：`js/pkg/auth/awscognito`）

3. 命名規則:

   - モジュール名は機能がわかるものにする、一部機能に分けにくいもの(複雑な presenter レイヤ等)だけ例外
   - インフラ層では機能とインフラ両方がわかる名前にする
   - 機能が明確なインフラ（例：mnist は手書き文字, elysia はサーバー）の場合は機能名を省略可能

4. レイヤー分離:

   - アプリケーション層とインフラ層は必ず別フォルダ（モジュール）に分ける
   - インフラ層はアプリケーション層で定義されたインターフェースに合わせて実装

5. import 構造:

   - インフラ層がモジュールを import する形
   - アプリケーション層は index.ts を使って、フォルダ単位で import 可能にする

6. 柔軟性:

   - 階層構造は状況に応じて柔軟に構成可能
   - ただし、基本的なレイヤー分離（アプリケーションとインフラ）は維持する

7. パッケージ公開:
   - package.json で適切にパスを公開する

8. その他: 
   - 固有名詞でなければinterface、固有名詞ならば実装
   - エラーメッセージは小文字から初めてピリオド不要

# TODO

今回考えた state machine を使った設計で、しかのこダイアグラムをポチポチ進められるページを作る

- camunda みたいなサービスを go で作りたい

1. 事前イベントと事後状態で状態遷移図を書く
2. 内部的に、事前イベントと事後状態の間に、中間イベントと事後イベントと取り消しイベントを挿入した状態遷移図に変換する
3. hook を用意して、事前イベントと処理を登録すると、処理を行って事後状態になるか、エラーを返して処理前の状態に戻る仕様を以下のように実装
   1. 事前イベントを発火して中間状態への遷移を試みる、無理だったら処理しないでエラーを返す
   2. 処理を実行する、処理に失敗したら取り消しイベントで事前イベント発火前の状態に戻る。エラーの定義は処理の中で行える
   3. 事後イベントを発火して事後状態へ遷移する、中間状態と事後イベントは自動生成されるので未定義のミスは発生せずかならず遷移できる

ターミナルからAI呼び出せる機能を作る
一つのターミナルにつき一つの会話
AIはAPI KeyではなくPlaywrightを使って行い無限に話せるようにする

# Memo

どんな設定しても vscode のターミナル上の venv がコマンドパレットからしか有効化できなかったので、ワークスペースで python を使う場合 terminal の venv の有効化はコマンドパレットから行う

なぜか mictlan から terminal を開かないと通知がでても venv は有効化されない

jupyter で python パッケージ利用できるようにするくだりかなり雑な設定になっているので注意がいる

poetry installとbun installは手動でやる、leanのダウンロードはエディタに言われてやる

# Note

- docker compose up(select services)はdevcontainerからできない、volumeマウントができない、ホストマシンから実行すべし
- Oracle.mysql-shell-for-vs-codeもバリ便利やけどdevcontainerから動かない、new windowしてホストからは見れる
- keybindindgs.jsonはdevcontainerではなくホストマシンの設定
