package qa

type Repo interface {
	AddQuestion(q Question)
	Answers(q Question) []Answer
}

type Storage interface {
}
