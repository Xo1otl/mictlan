# mictlan

monorepoは人生

[lean のワークスペースについて](https://github.com/leanprover/lean4/blob/master/src/lake/README.md)
なんか subpackage に manifest がないとか言われるが動いてるので問題なし

# パッケージ配置ルール

## 1. 柔軟なカテゴライゼーション

- 理解しやすさを重視し、自由度の高いカテゴライゼーションを許容する。
- 抽象的すぎる概念でのカテゴライゼーションは避ける。
- パッケージの入れ子構造を許可する（例：バックエンドパッケージ内にフロントエンドを含む）。
- 自分にとって意味のあるカテゴライズを考えて、適切な場所にpackageを配置した後、対応するpackage管理ツールで登録を行う。

## 2. マルチ言語対応

- 言語ごとにworkspace管理ツールが違う、packageを追加する時に言語ごとの管理ツールで登録する必要がある。
  - phpの場合は`workspace.php`
  - pythonの場合`pyproject.toml`
  - tsの場合`package.json`
  - goの場合`go.work`
  - leanの場合`lakefile.lean`
- 同一機能を異なる言語で実装する可能性を考慮する。
- 言語別のサブディレクトリを使用し、共通のパッケージ名を維持する。
  例：
  ```
  my-package/
  ├── go/
  │   └── (Goの実装)
  ├── python/
  │   └── (Pythonの実装)
  └── ts/
      └── (TypeScriptの実装)
  ```
- フォルダ名は必ずしもパッケージ名を反映しない。

## 3. 将来的なカテゴライゼーション

- 現在のプロジェクト構造は、npmのturborepoの慣例に倣って以下のように整理されています：
  - `apps/`: 完全なアプリケーション
  - `packages/`: 他のアプリケーションから呼び出される共有ライブラリ等
- プロジェクトの成長に応じて、より適切なカテゴライゼーションを検討する。
  （例：`packages/platforms`, `packages/sdks`, `services`, `frameworks`など）

## 注意事項

- カテゴライゼーションは定期的に見直し、必要に応じて調整する。
- 新しいパッケージを追加する際は、これらのルールに従いつつ、プロジェクト全体の一貫性を保つ。
- ドメイン駆動の分類を優先し、技術的な分類（言語やUIフレームワークなど）は二次的に考慮する。

# パッケージのルール

## 1. モジュールに分ける

- フロントエンドはレイヤー別
- バックエンドはドメイン別
- ライブラリやフレームワークやsdkは機能別等

## 2. Clean Architecture

- インフラ層はアプリケーション層で定義されたインターフェースに合わせて実装
- インフラ層がモジュールを import する形
- アプリケーション層は index.ts を使って、フォルダ単位で import 可能にする

## 3. パッケージ公開

- package.json で適切にパスを公開する
- project rootにある言語ごとのworkspaceファイルにpackageを登録

## 4. その他

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

- ターミナルからAI呼び出せる機能を作る、一つのターミナルにつき一つの会話
- AIはAPI KeyではなくPlaywrightを使って行い無限に話せるようにする
- ねずっちの謎掛けがAIで普通にできそうだからそれを行うサイト作ってみたい

# Memo

- leanのダウンロードとcomposerはエディタの通知に従ってやる
- 基本framework,driver層は自分で書かないけどuiは自作する唯一のinfraな気がする

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
