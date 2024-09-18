package qa

import (
	"fmt"
	"log"
	"net/http"
	"ossekaiserver/internal/auth"
	"regexp"

	"github.com/labstack/echo/v4"
)

func AddEchoRoutes(e *echo.Echo) {
	ch := NewCommandHandler()
	qh := NewQueryHandler()

	route := e.Group("/qa")
	route.Use(auth.EchoMiddleware())
	route.POST("/ask-question", ch.AskQuestion)
	route.GET("/find-tag", qh.FindTag)
}

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
	title, err := NewTitle(c.FormValue("title"))
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": err.Error()})
	}
	form, err := c.MultipartForm()
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": err.Error()})
	}
	tagSet, err := NewTagSet(form.Value["tag_ids"], form.Value["tag_names"])
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": err.Error()})
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
	objects := make([]*Object, 0, len(form.File["files"]))
	files := form.File["files"]
	for _, file := range files {
		src, err := file.Open()
		if err != nil {
			return c.JSON(http.StatusInternalServerError, map[string]string{"error": err.Error()})
		}
		object, err := NewObject(file.Filename, src)
		if err != nil {
			return c.JSON(http.StatusInternalServerError, map[string]string{"error": err.Error()})
		}
		objects = append(objects, object)
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
	questionId, err := h.command.AskQuestion(claims.(*auth.Claims).Sub, *title, tagSet, content)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": err.Error()})
	}
	return c.JSON(http.StatusOK, map[string]string{"questionId": string(*questionId)})
}

type QueryHandler struct {
	query *Query
}

func NewQueryHandler() *QueryHandler {
	repo := NewMockDbAsQueryRepo()
	query := NewQuery(repo)
	return &QueryHandler{query}
}

func (h *QueryHandler) FindTag(c echo.Context) error {
	claims := c.Get("claims")
	if claims == nil {
		return c.JSON(http.StatusUnauthorized, map[string]string{"error": "claims not found"})
	}
	name := c.QueryParam("name")
	if name != "" {
		tag, err := h.query.FindTagByName(name)
		if err != nil {
			return c.JSON(http.StatusOK, "{}")
		}
		return c.JSON(http.StatusOK, &TagDTO{Id: string(tag.Id), Name: tag.Name})
	}
	return c.JSON(http.StatusBadRequest, map[string]string{"error": "invalid query"})
}

type TagDTO struct {
	Id   string `json:"id"`
	Name string `json:"name"`
}
