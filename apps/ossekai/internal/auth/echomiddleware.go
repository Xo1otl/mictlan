package auth

import (
	"log"
	"net/http"
	"strings"

	"github.com/labstack/echo/v4"
)

func EchoMiddleware() echo.MiddlewareFunc {
	tokenService, err := NewAwsTokenService()
	if err != nil {
		log.Fatal(tokenService)
	}

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
			if err != nil {
				log.Fatal(err)
			}
			claims, err := NewClaims(&token, tokenService.Parse)
			if err != nil {
				return c.JSON(http.StatusUnauthorized, map[string]string{"error": "invalid token format"})
			}
			c.Set("claims", claims)
			return next(c)
		}
	}
}
