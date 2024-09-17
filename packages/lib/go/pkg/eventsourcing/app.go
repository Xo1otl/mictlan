package eventsourcing

import (
	"encoding/json"

	"github.com/google/uuid"
)

type EventRepo interface {
	Save(event *Event)
	BatchSave(event []*Event)
	// ここでのidはAggregateId
	Events(id uuid.UUID) []*Event
}

func CreateEvent(agg Aggregate, payload any) (*Event, error) {
	m := agg.MetaPtr()
	marshall, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	event := NewEvent(m.Id, m.Type, uuid.New(), m.State, m.SeqNum, marshall)
	return event, nil
}
