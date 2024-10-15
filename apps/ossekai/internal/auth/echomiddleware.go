package auth

import (
	"net/http"
	"strings"

	"github.com/labstack/echo/v4"
)

func EchoMiddleware(tokenService TokenService) echo.MiddlewareFunc {
	return func(next echo.HandlerFunc) echo.HandlerFunc {
		return func(c echo.Context) error {
			authHeader := c.Request().Header.Get("Authorization")
			if authHeader == "" {
				return c.JSON(http.StatusUnauthorized, map[string]string{"error": "authorization header is required"})
			}
			bearerToken := strings.Split(authHeader, " ")
			if len(bearerToken) != 2 {
				return c.JSON(http.StatusUnauthorized, map[string]string{"error": "invalid authorization header"})
			}
			token := Token(bearerToken[1])
			claims, err := NewClaims(&token, tokenService)
			if err != nil {
				return c.JSON(http.StatusUnauthorized, map[string]string{"error": "invalid token format"})
			}
			c.Set("claims", claims)
			return next(c)
		}
	}
}
