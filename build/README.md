# build

- 直接ビルドの設定を含んでいるファイルを置くところ
- 開発環境自体のビルドの内容に限定される、デプロイ含んでる場合はinfraに書く

## secrets.json

- すべてのシークレットファイルの参照を持っている
- secretmが作ったファイルで、ここを追記したり消したりすると管理するシークレットを編集できる
- 自動追加されるのでgitignoreからも消すこと推奨
