# OIDC

OAuth2.0 OIDCを自作したい

go言語とsvelteかreact19で作る



## APIServer

アクセストークン、IDトークン、リフレッシュトークンを発行

JWKS等で、公開鍵による検証を行う

client_secret, pkceに対応

LDAPサーバーを別で用意して問い合わせてログイン

TODO: RFCに従ったエンドポイントを列挙

## UI

clientの登録など
