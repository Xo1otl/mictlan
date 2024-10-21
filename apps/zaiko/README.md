# Zaiko

在庫管理システム

## Memo

- リモート`35.78.179.182`
- `CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build`でビルド
- infraで`generate_docker_compose.py`した後`deploy_awsjob.py`実行してからzaikoのyamlでportの1234を80にして`docker compose up redpanda redpanda-console mongo zaiko`
