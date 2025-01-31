# IdP

## Memo
* [Autheliaの設定](https://docs.ibracorp.io/authelia/configuration-files/configuration.yml)
* 設定のテスト: `docker run --rm --mount type=bind,source=./configuration.yml,target=/config/config.yml authelia/authelia authelia validate-config --config /config/config.yml`
