# build

- 複雑だったり依存のあるyamlはpythonで書いてシリアライズする
- scriptsのgen_tpl.py

## secrets.json

- すべてのシークレットファイルの参照を持っている
- secretmが作ったファイルで、ここを追記したり消したりすると管理するシークレットを編集できる
- 自動追加されるのでgitignoreからも消すこと推奨

## Memo

- dockerのincludeという文法がある
- サービスによっては複数のドメインで使われる場合がある
- その場合はドメイン固有の設定をドメインのフォルダに書いて、サービス自体のフォルダを別で用意してDIContainerのようにして集約する
- メインのdocker composeではその集約をincludeする
