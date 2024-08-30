package qa

type Repo interface {
	AskQuestion(q Question)
	Answers(q Question) []Answer
}
