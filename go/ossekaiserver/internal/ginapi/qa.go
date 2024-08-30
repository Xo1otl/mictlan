package ginapi

import (
	"log"
	"net/http"
	"ossekaiserver/internal/dynamodb"
	"ossekaiserver/internal/qa"

	"github.com/gin-gonic/gin"
)

func AddQARoutes(r *gin.Engine) {
	qaRepo := dynamodb.NewQARepo()
	qaApp := qa.NewApp(qaRepo)

	qaRoute := r.Group("/qa")

	qaRoute.Use(AuthMiddleware())

	qaRoute.POST("/ask-questions", func(c *gin.Context) {
		claims, _ := c.Get("claims")
		log.Printf("claims: %v", claims)
		q := qa.NewQuestion("")
		qaApp.AskQuestion(q)
	})
	qaRoute.POST("/answers", func(c *gin.Context) {
		claims, _ := c.Get("claims")
		log.Printf("claims: %v", claims)
		q := qa.NewQuestion("")
		qaAnswers := qaApp.Answers(q)

		answers := make([]gin.H, len(qaAnswers))
		for i, qaAnswer := range qaAnswers {
			answers[i] = gin.H{
				"sub": string(qaAnswer.Sub),
			}
		}

		c.JSON(http.StatusOK, gin.H{
			"answers": answers,
		})
	})
}
