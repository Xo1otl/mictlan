package qa

import (
	"fmt"
	"log"
	"net/http"
	"ossekaiserver/internal/auth"

	"github.com/labstack/echo/v4"
)

type AskQuestionInput struct {
	Title         string   `form:"title" binding:"required"`
	TagIds        []string `form:"tag_ids"`
	ContentBlocks []struct {
		Type    string `form:"type"`
		Content string `form:"content"`
	} `form:"content_blocks"`
}

type Handler struct {
	app *App
}

func NewHandler() *Handler {
	repo := NewMockDb()
	storage := NewMockStorage()
	app := NewApp(repo, storage)
	return &Handler{app}
}

func (h *Handler) AskQuestions(c echo.Context) error {
	claims := c.Get("claims")
	if claims == nil {
		return c.JSON(http.StatusUnauthorized, map[string]string{"error": "claims not found"})
	}
	title := c.FormValue("title")
	form, err := c.MultipartForm()
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": err.Error()})
	}
	var tagIds []TagId
	for _, tagId := range form.Value["tag_ids"] {
		tagIds = append(tagIds, TagId(tagId))
	}
	log.Printf("tagIds: %v", tagIds)
	var contentBlocks []*ContentBlock
	for i := 0; ; i++ {
		typeKey := fmt.Sprintf("contentBlocks[%d][type]", i)
		contentKey := fmt.Sprintf("contentBlocks[%d][content]", i)
		blockType := c.FormValue(typeKey)
		content := c.FormValue(contentKey)
		if blockType == "" && content == "" {
			break // これ以上のブロックがない場合
		}
		contentBlock, err := NewContentBlock(blockType, content)
		if err != nil {
			return c.JSON(http.StatusInternalServerError, map[string]string{"error": err.Error()})
		}
		contentBlocks = append(contentBlocks, contentBlock)
	}
	objects := make([]*Object, len(form.File["files"]))
	files := form.File["files"]
	for i, file := range files {
		src, err := file.Open()
		if err != nil {
			return c.JSON(http.StatusInternalServerError, map[string]string{"error": err.Error()})
		}
		object, err := NewObject(file.Filename, src)
		if err != nil {
			return c.JSON(http.StatusInternalServerError, map[string]string{"error": err.Error()})
		}
		objects[i] = object
	}
	log.Printf("files: %v", files)
	questionId, err := h.app.AskQuestion(claims.(auth.Claims).Sub, title, tagIds, contentBlocks, objects)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": err.Error()})
	}
	return c.JSON(http.StatusOK, map[string]string{"questionId": string(*questionId)})
}

func (h *Handler) Answers(c echo.Context) error {
	claims := c.Get("claims")
	if claims == nil {
		return c.JSON(http.StatusUnauthorized, map[string]string{"error": "claims not found"})
	}
	log.Printf("claims: %v", claims)
	q := Question{}
	qaAnswers := h.app.Answers(q.Id)

	answers := make([]map[string]string, len(qaAnswers))
	for i, qaAnswer := range qaAnswers {
		answers[i] = map[string]string{
			"sub": string(qaAnswer.Sub),
		}
	}

	return c.JSON(http.StatusOK, map[string]interface{}{
		"answers": answers,
	})
}

func AddEchoRoutes(e *echo.Echo) {
	h := NewHandler()

	route := e.Group("/qa")
	route.Use(auth.EchoMiddleware())
	route.POST("/ask-questions", h.AskQuestions)
	route.POST("/answers", h.Answers)
}
