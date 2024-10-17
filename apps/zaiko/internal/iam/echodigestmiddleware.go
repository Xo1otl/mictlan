package iam

import (
	"net/http"

	"github.com/labstack/echo/v4"
)

// EchoDigestMiddlewareはDigest認証用のミドルウェアです
func EchoDigestMiddleware(digest *Digest) echo.MiddlewareFunc {
	return func(next echo.HandlerFunc) echo.HandlerFunc {
		return func(c echo.Context) error {
			authHeader := c.Request().Header.Get("Authorization")

			// Authorizationヘッダーが存在しない場合
			if authHeader == "" {
				// Digest認証の初期化（nonceやopaqueの生成）
				realm := "localhost" // 固定のrealm。動的に変更することもできます。
				digestToken := digest.Init(realm)

				// ダイジェスト認証を要求するWWW-Authenticateヘッダーを動的に生成
				wwwAuthenticateHeader := `Digest realm="` + digestToken.Realm + `", qop="` + digestToken.Qop + `", nonce="` + digestToken.Nonce + `", opaque="` + digestToken.Opaque + `"`

				// クライアントに認証を要求するヘッダーを追加
				c.Response().Header().Set("WWW-Authenticate", wwwAuthenticateHeader)
				return c.JSON(http.StatusUnauthorized, map[string]string{"error": "authorization header is required"})
			}

			// Authorizationヘッダーがある場合は、次のハンドラーへ
			return next(c)
		}
	}
}
