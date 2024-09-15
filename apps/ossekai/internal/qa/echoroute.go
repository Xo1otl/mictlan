package qa

import (
	"fmt"
	"log"
	"net/http"
	"ossekaiserver/internal/auth"
	"regexp"

	"github.com/labstack/echo/v4"
)

type CommandHandler struct {
	command *Command
}

func NewCommandHandler() *CommandHandler {
	repo := NewMockDb()
	storage := NewMockStorage()
	command := NewCommand(repo, storage)
	return &CommandHandler{command}
}

func (h *CommandHandler) AskQuestion(c echo.Context) error {
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
	parse := func(s string) ([]string, error) {
		// Define the regular expression to match placeholders
		re, err := regexp.Compile(`!\[([^\]]+)\]`)
		if err != nil {
			return nil, err
		}

		// Find all matches in the input string
		matches := re.FindAllStringSubmatch(s, -1)

		// Extract the placeholders from the matches
		var placeholders []string
		for _, match := range matches {
			if len(match) > 1 {
				placeholders = append(placeholders, match[1])
			}
		}

		return placeholders, nil
	}
	content, err := NewContent(contentBlocks, objects, parse)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": err.Error()})
	}
	questionId, err := h.command.AskQuestion(claims.(*auth.Claims).Sub, title, tagNames, content)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": err.Error()})
	}
	return c.JSON(http.StatusOK, map[string]string{"questionId": string(*questionId)})
}

func AddEchoRoutes(e *echo.Echo) {
	mh := NewCommandHandler()

	route := e.Group("/qa")
	route.Use(auth.EchoMiddleware())
	route.POST("/ask-question", mh.AskQuestion)
}
