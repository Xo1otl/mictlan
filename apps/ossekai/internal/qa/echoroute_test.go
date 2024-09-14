package qa_test

import (
	"bytes"
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
	req := httptest.NewRequest(http.MethodPost, "/qa/answers", body)
	randomAuth := make([]byte, 32)
	gofakeit.Slice(&randomAuth)
	req.Header.Set("Content-Type", contentType)
	rec := httptest.NewRecorder()
	c := e.NewContext(req, rec)
	c.Set("claims", auth.Claims{Sub: auth.Sub(gofakeit.UUID())})

	h := qa.NewMutationHandler()
	err = h.AskQuestions(c)
	t.Log(rec.Body.String())
	t.Log(err)
}

func generateMultipartBody() (*bytes.Buffer, string, error) {
	body := new(bytes.Buffer)
	writer := multipart.NewWriter(body)

	// Add title
	err := writer.WriteField("title", gofakeit.Sentence(5))
	if err != nil {
		return nil, "", err
	}

	// Add multiple tag_ids
	tagCount := gofakeit.Number(1, 5)
	for i := 0; i < tagCount; i++ {
		err := writer.WriteField("tag_ids", gofakeit.ProgrammingLanguage())
		if err != nil {
			return nil, "", err
		}
	}

	// Add multiple contentBlocks with type and content
	contentBlockCount := gofakeit.Number(1, 5)
	for i := 0; i < contentBlockCount; i++ {
		kind := gofakeit.RandomString([]string{"text", "latex", "markdown"})
		var content string
		switch kind {
		case "text":
			content = gofakeit.Paragraph(1, 3, 10, "\n")
		case "latex":
			content = gofakeit.LoremIpsumSentence(50)
		case "markdown":
			content = gofakeit.LoremIpsumSentence(50)
		}

		err := writer.WriteField(fmt.Sprintf("contentBlocks[%d][kind]", i), kind)
		if err != nil {
			return nil, "", err
		}
		err = writer.WriteField(fmt.Sprintf("contentBlocks[%d][content]", i), content)
		if err != nil {
			return nil, "", err
		}
	}

	// Add multiple files
	fileCount := gofakeit.Number(1, 3)
	for i := 0; i < fileCount; i++ {
		fileName := gofakeit.AppName()
		part, err := writer.CreateFormFile("files", fileName)
		if err != nil {
			return nil, "", err
		}
		_, err = part.Write([]byte(gofakeit.LoremIpsumParagraph(1, 5, 10, "\n")))
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
