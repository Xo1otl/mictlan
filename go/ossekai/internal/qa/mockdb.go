package qa

import (
	"github.com/brianvoe/gofakeit/v7"
)

type MockDb struct{}

func (m *MockDb) Answers(q Question) []Answer {
	count := 10
	answers := make([]Answer, count)
	gofakeit.Slice(&answers)
	return answers
}

func (m *MockDb) AddQuestion(q Question) {
	panic("unimplemented")
}

func NewMockDb() Repo {
	return &MockDb{}
}
