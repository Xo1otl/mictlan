package eventsourcing

import (
	"errors"

	"github.com/google/uuid"
)

var ErrInvalidSeqNum = errors.New("SeqNum are not consecutive")

type AggregateMeta struct {
	Id     uuid.UUID
	Type   string
	State  string
	SeqNum int
}

func (m *AggregateMeta) MetaPtr() *AggregateMeta {
	return m
}

func (m *AggregateMeta) Update(state string) {
	m.State = state
	m.SeqNum++
}

func NewAggregateMeta() *AggregateMeta {
	return &AggregateMeta{}
}

type Aggregate interface {
	MetaPtr() *AggregateMeta
	Apply(event *Event) error
}

func ReplayEvents(agg Aggregate, events []*Event) error {
	for _, event := range events {
		if event.SeqNum-agg.MetaPtr().SeqNum != 1 {
			return ErrInvalidSeqNum
		}
		err := agg.Apply(event)
		if err != nil {
			return err
		}
	}
	return nil
}

func InitAggregate(agg Aggregate, events []*Event, aggregateType, initialState string) error {
	agg.MetaPtr().Type = aggregateType
	agg.MetaPtr().State = initialState
	if events == nil {
		agg.MetaPtr().Id = uuid.New()
	} else {
		agg.MetaPtr().Id = events[0].AggregateId
	}
	err := ReplayEvents(agg, events)
	return err
}
