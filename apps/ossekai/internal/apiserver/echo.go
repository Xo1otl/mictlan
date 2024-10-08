package apiserver

import (
	"log"
	"ossekaiserver/internal/auth"
	"ossekaiserver/internal/qa"

	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
)

func LaunchEcho() {
	e := echo.New()

	// CORS configuration
	e.Use(middleware.CORSWithConfig(middleware.CORSConfig{
		AllowOrigins:     []string{"http://localhost:5173"},
		AllowHeaders:     []string{echo.HeaderOrigin, echo.HeaderContentType, echo.HeaderAccept, "Authorization"},
		AllowMethods:     []string{echo.GET, echo.POST, echo.PUT, echo.PATCH, echo.DELETE, echo.HEAD, echo.OPTIONS},
		AllowCredentials: true,
	}))

	// Use custom authentication middleware
	e.Use(auth.EchoMiddleware())

	// Add routes from qa package
	qa.AddEchoRoutes(e)

	log.Println("🚀 Server listening at: http://localhost:3030")
	err := e.Start("localhost:3030")
	if err != nil {
		log.Fatal(err)
	}
}
