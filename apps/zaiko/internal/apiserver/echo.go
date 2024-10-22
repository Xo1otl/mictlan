package apiserver

import (
	"log"
	"zaiko/internal/auth"
	"zaiko/internal/iam"
	"zaiko/internal/stock"

	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
)

func LaunchEcho(addr string) {
	e := echo.New()

	// CORS configuration
	e.Use(middleware.CORSWithConfig(middleware.CORSConfig{
		AllowOrigins:     []string{"http://localhost:5173"},
		AllowHeaders:     []string{echo.HeaderOrigin, echo.HeaderContentType, echo.HeaderAccept, "Authorization"},
		AllowMethods:     []string{echo.GET, echo.POST, echo.PUT, echo.PATCH, echo.DELETE, echo.HEAD, echo.OPTIONS},
		AllowCredentials: true,
	}))
	e.Use(middleware.Logger())

	// 課題1
	e.GET("/", func(c echo.Context) error {
		return c.String(200, "AWS")
	})

	// 課題2
	accountRepo := iam.NewInMemoryAccountRepo()
	digestNcRepo := iam.NewInMemoryDigestNcRepo()
	validator := iam.NewMD5DigestValidator()
	nonceServce := iam.NewHMACNonceService("demo_secert")
	tokenService := iam.NewDigest(accountRepo, digestNcRepo, nonceServce, validator)

	// secret routeには認証用のミドルウェアを追加
	e.GET("/secret", func(c echo.Context) error {
		log.Println("Secret route accessed")
		return c.String(200, "SUCCESS")
	}, iam.EchoDigestMiddleware(tokenService), auth.EchoMiddleware(tokenService))

	// 課題3
	stock.AddEchoRoutes(e)

	log.Println("🚀 Server listening at: http://" + addr)
	err := e.Start(addr)
	if err != nil {
		log.Fatal(err)
	}
}
