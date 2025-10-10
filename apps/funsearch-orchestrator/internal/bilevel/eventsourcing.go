package bilevel

type CallType string

const (
	CallIssue  CallType = "Issue"
	CallUpdate CallType = "Update"
)

type StateEvent[ORes any] struct {
	Type    CallType
	Payload ORes // nil if Type is CallIssue
}

func WithEventSourcing[PReq, ORes any](s State[PReq, ORes]) (State[PReq, ORes], func() []StateEvent[ORes]) {
	wrapper := &esState[PReq, ORes]{State: s}
	return wrapper, func() []StateEvent[ORes] {
		return wrapper.events
	}
}

func Replay[PReq, ORes any](State State[PReq, ORes], trace []StateEvent[ORes]) {
	for _, event := range trace {
		switch event.Type {
		case CallIssue:
			State.Issue()
		case CallUpdate:
			State.Update(event.Payload)
		}
	}
}

type esState[PReq, ORes any] struct {
	State[PReq, ORes]
	events []StateEvent[ORes]
}

func (s *esState[PReq, ORes]) Update(res ORes) (done bool, err error) {
	s.events = append(s.events, StateEvent[ORes]{Type: CallUpdate, Payload: res})
	return s.State.Update(res)
}

func (s *esState[PReq, ORes]) Issue() (req PReq, ok bool, err error) {
	s.events = append(s.events, StateEvent[ORes]{Type: CallIssue})
	return s.State.Issue()
}
