# infra

infrastructure as code

基本pythonで書くこととする

## Memo

- サービスによっては複数のドメインで使われる場合がある
- その場合はドメイン固有の設定をドメインのフォルダに書いて、サービス自体のフォルダを別で用意してDIContainerのようにして集約する
- メインのdocker composeではその集約をincludeする
- pythonパッケージとして登録されているのでimportが使える
