package qa

import (
	"ossekaiserver/internal/auth"
	"time"
)

type Tag struct{}
type QuestionId string
type QuestionMeta struct {
	Sub          auth.Sub
	Id           QuestionId
	CreatedAt    time.Time
	UpdatedAt    time.Time
	Solved       bool
	BestAnswerId AnswerId
	Title        string
	Tags         []Tag
}
type QuestionContent struct {
	Text string
	// TODO: Attachmentや数式
}

type Question struct {
	QuestionMeta
	QuestionContent
}

func NewQuestion(sub auth.Sub) Question {
	return Question{}
}

type AnswerId string
type Answer struct {
	Sub auth.Sub
	Id  AnswerId
}

func NewAnswer(sub auth.Sub) Answer {
	return Answer{Sub: sub}
}
