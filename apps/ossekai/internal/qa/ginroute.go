package qa

import (
	"log"
	"net/http"
	"ossekaiserver/internal/auth"

	"github.com/gin-gonic/gin"
)

// type AskQuestionInput struct {
// 	Title         string   `form:"title" binding:"required"`
// 	TagIds        []string `form:"tag_ids"`
// 	ContentBlocks []struct {
// 		Type    string `form:"type"`
// 		Content string `form:"content"`
// 	} `form:"content_blocks"`
// }

func AddGinRoutes(r *gin.Engine) {
	repo := NewMockDb()
	storage := NewMockStorage()
	app := NewApp(repo, storage)

	route := r.Group("/qa")

	route.Use(auth.GinMiddleware())

	route.POST("/ask-questions", func(c *gin.Context) {
		claims, exists := c.Get("claims")
		if !exists {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "claims not found"})
			return
		}
		log.Printf("claims: %v", claims)
		c.JSON(http.StatusOK, gin.H{"status": "question asked"})
	})

	route.POST("/answers", func(c *gin.Context) {
		claims, exists := c.Get("claims")
		if !exists {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "claims not found"})
			return
		}
		claims = claims.(auth.Claims)
		log.Printf("claims: %v", claims)
		q := Question{}
		qaAnswers := app.Answers(q.Id)

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
