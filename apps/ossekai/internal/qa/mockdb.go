package qa

import (
	"ossekaiserver/internal/auth"

	"github.com/brianvoe/gofakeit/v7"
)

type MockDb struct{}

// AddQuestion implements Repo.
func (m *MockDb) AddQuestion(sub auth.Sub, title string, tagIds []TagId, contentBlocks []ContentBlock, attachments Attachments) (*QuestionId, error) {
	panic("unimplemented")
}

// Answers implements Repo.
func (m *MockDb) Answers(questionId QuestionId) []Answer {
	count := 10
	answers := make([]Answer, count)
	gofakeit.Slice(&answers)
	return answers
}

// Question implements Repo.
func (m *MockDb) Question(questionId QuestionId) Question {
	panic("unimplemented")
}

func NewMockDb() Repo {
	return &MockDb{}
}
