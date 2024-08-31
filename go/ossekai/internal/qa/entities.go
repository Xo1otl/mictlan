package qa

import "ossekaiserver/internal/auth"

type Question struct {
	Sub auth.Sub
}

func NewQuestion(sub auth.Sub) Question {
	return Question{sub}
}

type Answer struct {
	Sub auth.Sub
}

func NewAnswer(sub auth.Sub) Answer {
	return Answer{sub}
}
