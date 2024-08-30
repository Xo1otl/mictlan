package dynamodb_test

import (
	"ossekaiserver/internal/qa"
	"testing"

	"github.com/brianvoe/gofakeit/v7"
)

func TestQARepo(t *testing.T) {
	answers := make([]qa.Answer, 10)
	gofakeit.Slice(&answers)
	t.Log(answers)
}
