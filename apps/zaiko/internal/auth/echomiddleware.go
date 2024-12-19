package auth

import (
	"log"
	"net/http"

	"github.com/labstack/echo/v4"
)

func EchoMiddleware(tokenService TokenService) echo.MiddlewareFunc {
	return func(next echo.HandlerFunc) echo.HandlerFunc {
		return func(c echo.Context) error {
			authHeader := c.Request().Header.Get("Authorization")

			// Authorizationヘッダーが存在しない場合
			if authHeader == "" {
				return c.JSON(http.StatusUnauthorized, map[string]string{"error": "authorization header is required"})
			}

			token := Token(authHeader)

			log.Println("Token parsed successfully")
			// トークンのバリデーションとパース
			claims, err := NewClaims(&token, tokenService)
			if err != nil {
				return c.JSON(http.StatusUnauthorized, map[string]string{"error": err.Error()})
			}

			// クレーム情報をコンテキストにセット
			c.Set("claims", claims)

			// 次のハンドラに処理を渡す
			return next(c)
		}
	}
}
