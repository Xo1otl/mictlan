package qa

import (
	"github.com/brianvoe/gofakeit/v7"
)

type DynamoDb struct{}

func (*DynamoDb) Answers(q Question) []Answer {
	count := 10
	answers := make([]Answer, count)
	gofakeit.Slice(&answers)
	return answers
}

func (*DynamoDb) AskQuestion(q Question) {
	panic("unimplemented")
}

func NewDynamoDb() Repo {
	return &DynamoDb{}
}
