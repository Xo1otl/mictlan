package qa

import "ossekaiserver/internal/auth"

type AnswerId string
type Answer struct {
	Sub auth.Sub
	Id  AnswerId
}

func NewAnswer(sub auth.Sub) Answer {
	return Answer{Sub: sub}
}
