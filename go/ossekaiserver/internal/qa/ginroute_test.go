package qa_test

import (
	"net/http"
	"net/http/httptest"
	"ossekaiserver/internal/qa"
	"testing"

	"github.com/brianvoe/gofakeit/v7"
	"github.com/gin-gonic/gin"
)

func TestGinRoutes(t *testing.T) {
	// テスト用のGinルーターを設定
	gin.SetMode(gin.TestMode)
	r := gin.New()
	qa.AddGinRoutes(r)

	// /qa/answers のテスト
	t.Run("Answers", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/qa/answers", nil)
		r.ServeHTTP(w, req)

		if w.Code != http.StatusUnauthorized {
			t.Errorf("Expected status 401, got %d", w.Code)
		}

		// ランダムな認証ヘッダーを生成
		randomAuth := make([]byte, 32)
		gofakeit.Slice(&randomAuth)
		randomAuthHeader := "Bearer " + string(randomAuth)

		req, _ = http.NewRequest("POST", "/qa/answers", nil)
		req.Header.Set("Authorization", randomAuthHeader)
		w = httptest.NewRecorder()
		r.ServeHTTP(w, req)

		if w.Code != http.StatusUnauthorized {
			t.Errorf("Expected status 401 with random auth header, got %d", w.Code)
		}
	})
}
