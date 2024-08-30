package dynamodb

import (
	"ossekaiserver/internal/qa"

	"github.com/brianvoe/gofakeit/v7"
)

type QARepo struct{}

func (*QARepo) Answers(q qa.Question) []qa.Answer {
	count := 10
	answers := make([]qa.Answer, count)
	gofakeit.Slice(&answers)
	return answers
}

func (*QARepo) AskQuestion(q qa.Question) {
	panic("unimplemented")
}

func NewQARepo() qa.Repo {
	return &QARepo{}
}
