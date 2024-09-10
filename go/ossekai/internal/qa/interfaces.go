package qa

type Repo interface {
	AddQuestion(q Question)
	Answers(q Question) []Answer
}

type Storage interface {
	// storageKeyを返す
	Put(date []byte) (string, error)
}
