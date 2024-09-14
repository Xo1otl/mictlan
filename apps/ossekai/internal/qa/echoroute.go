package qa

import (
	"fmt"
	"log"
	"net/http"
	"ossekaiserver/internal/auth"

	"github.com/labstack/echo/v4"
)

type MutationHandler struct {
	mutation *Mutation
}

func NewMutationHandler() *MutationHandler {
	repo := NewMockDb()
	storage := NewMockStorage()
	mutation := NewMutation(repo, storage)
	return &MutationHandler{mutation}
}

func (h *MutationHandler) AskQuestions(c echo.Context) error {
	claims := c.Get("claims")
	if claims == nil {
		return c.JSON(http.StatusUnauthorized, map[string]string{"error": "claims not found"})
	}
	title := c.FormValue("title")
	form, err := c.MultipartForm()
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": err.Error()})
	}
	var tagNames []TagName
	for _, tagId := range form.Value["tag_ids"] {
		tagNames = append(tagNames, TagName(tagId))
	}
	var contentBlocks []*ContentBlock
	for i := 0; ; i++ {
		kindKey := fmt.Sprintf("contentBlocks[%d][kind]", i)
		contentKey := fmt.Sprintf("contentBlocks[%d][content]", i)
		blockKind := c.FormValue(kindKey)
		content := c.FormValue(contentKey)
		if blockKind == "" && content == "" {
			break // これ以上のブロックがない場合
		}
		contentBlock, err := NewContentBlock(blockKind, content)
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
	questionId, err := h.mutation.AskQuestion(claims.(auth.Claims).Sub, title, tagNames, contentBlocks, objects)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": err.Error()})
	}
	return c.JSON(http.StatusOK, map[string]string{"questionId": string(*questionId)})
}

func AddEchoRoutes(e *echo.Echo) {
	mh := NewMutationHandler()

	route := e.Group("/qa")
	route.Use(auth.EchoMiddleware())
	route.POST("/ask-questions", mh.AskQuestions)
}
