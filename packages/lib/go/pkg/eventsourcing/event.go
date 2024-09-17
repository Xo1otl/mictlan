package eventsourcing

import "github.com/google/uuid"

type Event struct {
	AggregateId   uuid.UUID
	AggregateType string
	EventId       uuid.UUID
	EventType     string
	SeqNum        int
	Payload       []byte
}

func NewEvent(
	aggregateId uuid.UUID,
	aggregateType string,
	eventId uuid.UUID,
	eventType string,
	seqNum int,
	payload []byte,
) *Event {
	return &Event{
		aggregateId,
		aggregateType,
		eventId,
		eventType,
		seqNum,
		payload,
	}
}
