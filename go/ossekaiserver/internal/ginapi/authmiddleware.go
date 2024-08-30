package ginapi

import (
	"log"
	"net/http"
	"ossekaiserver/internal/auth"
	"ossekaiserver/internal/auth/cognitojwt"
	"strings"

	"github.com/gin-gonic/gin"
)

func AuthMiddleware() gin.HandlerFunc {
	tokenService, err := cognitojwt.NewTokenService()
	if err != nil {
		log.Fatal(tokenService)
	}

	return func(c *gin.Context) {
		authHeader := c.GetHeader("Authorization")
		if authHeader == "" {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "authorization header is required"})
			c.Abort()
			return
		}
		bearerToken := strings.Split(authHeader, " ")
		if len(bearerToken) != 2 {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "invalid authorization header"})
			c.Abort()
			return
		}
		token := auth.Token(bearerToken[1])
		if err != nil {
			log.Fatal(err)
		}
		claims, err := auth.NewClaims(&token, tokenService.Parse)
		if err != nil {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "invalid token format"})
			c.Abort()
			return
		}
		c.Set("claims", claims)
		c.Next()
	}
}
