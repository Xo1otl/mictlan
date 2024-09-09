package qa

type Repo interface {
	AskQuestion(q Question)
	Answers(q Question) []Answer
}

type Storage interface {
	Store(data []byte) error
	Retrieve() ([]byte, error)
	Delete() error
}
