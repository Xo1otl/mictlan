# Zaiko

在庫管理システム

## Memo

- `CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build`でビルド
- infraで`generate_docker_compose.py`した後`deploy_awsjob.py`実行してからzaikoのyamlでportの1234を80にして`docker compose up redpanda redpanda-console mongo zaiko`
    - zaikoだけredpandaの立ち上げを待つ必要がある。そうしないとechoserverがエラーで終了する。修正かヘルスチェック的なものを検討する。
- 最初のイベントのproduceだけ少し時間がかかるっぽい。スクリプトでやると必ず反映されておらずテストに失敗する。
