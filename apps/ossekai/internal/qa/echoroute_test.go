package qa_test

import (
	"bytes"
	"fmt"
	"log"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"ossekaiserver/internal/auth"
	"ossekaiserver/internal/qa"
	"testing"

	"github.com/brianvoe/gofakeit/v7"
	"github.com/labstack/echo/v4"
)

func TestEchoRoute(t *testing.T) {
	e := echo.New()
	body, contentType, err := generateMultipartBody()
	if err != nil {
		t.Errorf("Failed to generate multipart body: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/qa/answers", body)
	randomAuth := make([]byte, 32)
	gofakeit.Slice(&randomAuth)
	req.Header.Set("Content-Type", contentType)
	rec := httptest.NewRecorder()
	c := e.NewContext(req, rec)
	c.Set("claims", auth.Claims{Sub: "sub"})

	h := qa.NewHandler()
	err = h.AskQuestions(c)
	log.Print(rec.Body.String())
	log.Print(err)
}

func generateMultipartBody() (*bytes.Buffer, string, error) {
	body := new(bytes.Buffer)
	writer := multipart.NewWriter(body)

	// Add title
	err := writer.WriteField("title", "Sample Title")
	if err != nil {
		return nil, "", err
	}

	// Add multiple tag_ids
	tagIDs := []string{"1", "2", "3"}
	for _, id := range tagIDs {
		err := writer.WriteField("tag_ids", id)
		if err != nil {
			return nil, "", err
		}
	}

	// Add multiple contentBlocks with type and content
	contentBlocks := []struct {
		Type    string
		Content string
	}{
		{"text", "This is a text block"},
		{"image", "path/to/image.jpg"},
		{"code", "fmt.Println(\"Hello, World!\")"},
	}

	for i, block := range contentBlocks {
		err := writer.WriteField(fmt.Sprintf("contentBlocks[%d][type]", i), block.Type)
		if err != nil {
			return nil, "", err
		}
		err = writer.WriteField(fmt.Sprintf("contentBlocks[%d][content]", i), block.Content)
		if err != nil {
			return nil, "", err
		}
	}

	// Add multiple files
	fileContents := []struct {
		name    string
		content string
	}{
		{"test1.txt", "This is the first test file"},
		{"test2.txt", "This is the second test file"},
		{"test3.txt", "This is the third test file"},
	}

	for _, file := range fileContents {
		part, err := writer.CreateFormFile("files", file.name)
		if err != nil {
			return nil, "", err
		}
		_, err = part.Write([]byte(file.content))
		if err != nil {
			return nil, "", err
		}
	}

	err = writer.Close()
	if err != nil {
		return nil, "", err
	}

	return body, writer.FormDataContentType(), nil
}
