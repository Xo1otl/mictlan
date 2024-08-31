package apiserver

import (
	"log"
	"ossekaiserver/internal/qa"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

func LaunchGin() {
	r := gin.Default()

	config := cors.DefaultConfig()
	config.AllowOrigins = []string{"http://localhost:5173"}
	config.AllowHeaders = append(config.AllowHeaders, "Authorization")
	config.AllowMethods = []string{"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
	config.AllowCredentials = true

	r.Use(cors.New(config))

	qa.AddGinRoutes(r)
	log.Println("🚀 Server listening at: http://localhost:3000")
	err := r.Run("localhost:3000")
	if err != nil {
		log.Fatal(err)
	}
}
