package qa_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"ossekaiserver/internal/auth"
	"ossekaiserver/internal/qa"
	"testing"

	"github.com/brianvoe/gofakeit/v7"
	"github.com/labstack/echo/v4"
)

func TestFindTagByName(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Test panicked: %v", r)
		}
	}()
	e := echo.New()
	req := httptest.NewRequest(http.MethodGet, "/qa/find-tag?name=AAAAAAAAAAAAAAAAAAAAA", nil)
	randomAuth := make([]byte, 32)
	gofakeit.Slice(&randomAuth)
	rec := httptest.NewRecorder()
	c := e.NewContext(req, rec)
	c.Set("claims", &auth.Claims{Sub: auth.Sub(gofakeit.UUID())})

	h := qa.NewQueryHandler()
	err := h.FindTag(c)
	if err != nil {
		t.Errorf("Failed to ask questions: %v", err)
	}
	t.Log(rec.Body.String())
}

func TestAskQuestion(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Test panicked: %v", r)
		}
	}()
	e := echo.New()
	body, contentType, err := generateMultipartBody()
	if err != nil {
		t.Errorf("Failed to generate multipart body: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/qa/ask-question", body)
	randomAuth := make([]byte, 32)
	gofakeit.Slice(&randomAuth)
	req.Header.Set("Content-Type", contentType)
	rec := httptest.NewRecorder()
	c := e.NewContext(req, rec)
	c.Set("claims", &auth.Claims{Sub: auth.Sub(gofakeit.UUID())})

	h := qa.NewCommandHandler()
	err = h.AskQuestion(c)
	if err != nil {
		t.Errorf("Failed to ask questions: %v", err)
	}
	var responseBody struct {
		Error string `json:"error"`
	}
	err = json.Unmarshal(rec.Body.Bytes(), &responseBody)
	if err != nil {
		t.Errorf("Failed to unmarshal response body: %v", err)
	}
	if responseBody.Error != "" {
		t.Errorf("test failed: %v", responseBody.Error)
	}
}

func generateMultipartBody() (*bytes.Buffer, string, error) {
	body := new(bytes.Buffer)
	writer := multipart.NewWriter(body)

	// Add title
	writer.WriteField("title", gofakeit.Sentence(5))

	// Add tag_ids
	for i := 0; i < gofakeit.Number(1, 5); i++ {
		writer.WriteField("tag_names", gofakeit.ProgrammingLanguage())
	}

	// Generate file names
	fileNames := make([]string, gofakeit.Number(1, 5))
	for i := range fileNames {
		fileNames[i] = gofakeit.AppName()
	}

	// Prepare contentBlocks
	contentBlockCount := gofakeit.Number(1, 5)
	contentBlocks := make([]struct {
		kind    string
		content string
	}, contentBlockCount)

	// Generate initial content for contentBlocks
	for i := 0; i < contentBlockCount; i++ {
		contentBlocks[i].kind = gofakeit.RandomString([]string{"text", "latex", "markdown"})
		contentBlocks[i].content = gofakeit.Paragraph(1, 3, 10, "\n")
	}

	// Add placeholders randomly
	for _, fileName := range fileNames {
		randomIndex := gofakeit.Number(0, contentBlockCount-1)
		placeholder := fmt.Sprintf("![%s]", fileName)
		contentBlocks[randomIndex].content += " " + placeholder
	}

	// Write contentBlocks to multipart writer
	for i, block := range contentBlocks {
		writer.WriteField(fmt.Sprintf("contentBlocks[%d][kind]", i), block.kind)
		writer.WriteField(fmt.Sprintf("contentBlocks[%d][content]", i), block.content)
	}

	// Add files
	for _, fileName := range fileNames {
		part, _ := writer.CreateFormFile("files", fileName)
		part.Write([]byte(gofakeit.Paragraph(1, 1, 5, "")))
	}

	return body, writer.FormDataContentType(), writer.Close()
}
