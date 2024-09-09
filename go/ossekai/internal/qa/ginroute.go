package qa

import (
	"log"
	"net/http"
	"ossekaiserver/internal/auth"

	"github.com/gin-gonic/gin"
)

func AddGinRoutes(r *gin.Engine) {
	repo := NewDynamoDb()
	storage := NewMockStorage()
	app := NewApp(repo, storage)

	route := r.Group("/qa")

	route.Use(auth.GinMiddleware())

	route.POST("/ask-questions", func(c *gin.Context) {
		claims, _ := c.Get("claims")
		log.Printf("claims: %v", claims)
		q := Question{}
		app.AskQuestion(q)
	})
	route.POST("/answers", func(c *gin.Context) {
		claims, _ := c.Get("claims")
		log.Printf("claims: %v", claims)
		q := Question{}
		qaAnswers := app.Answers(q)

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
