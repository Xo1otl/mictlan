package stock

import (
	"zaiko/internal/auth"

	"github.com/shopspring/decimal"
)

// Commandのほうで使うEventConsumer/Producerの実装
type InMemoryEventStore struct {
	events           []any
	projectionEvents []AggregateUpdatedEvent
}

func (s *InMemoryEventStore) Events(sub auth.Sub) ([]any, error) {
	return s.events, nil
}

func (s *InMemoryEventStore) OnAdded(event AddedEvent) error {
	s.events = append(s.events, event)
	return nil
}

func (s *InMemoryEventStore) OnSold(event SoldEvent) error {
	s.events = append(s.events, event)
	return nil
}

func (s *InMemoryEventStore) OnClearedAll(event ClearedAllEvent) error {
	s.events = append(s.events, event)
	return nil
}

func (s *InMemoryEventStore) OnAggregateUpdated(event AggregateUpdatedEvent) error {
	s.projectionEvents = append(s.projectionEvents, event)
	return nil
}

// Queryのほうで使うrepositoryの実装
type InMemoryRepo struct {
	*InMemoryEventStore
}

func NewInMemoryRepo(store *InMemoryEventStore) Repo {
	return &InMemoryRepo{store}
}

func (r *InMemoryRepo) Stocks(name string) map[string]int {
	if len(r.projectionEvents) == 0 {
		return map[string]int{}
	}
	if name == "" {
		return r.projectionEvents[len(r.projectionEvents)-1].Stocks
	}
	stockValue := r.projectionEvents[len(r.projectionEvents)-1].Stocks[name]
	return map[string]int{name: stockValue}
}

func (r *InMemoryRepo) Sales() decimal.Decimal {
	if len(r.projectionEvents) == 0 {
		return decimal.Zero // r.projectionEventsが空の場合、0を返す
	}
	return r.projectionEvents[len(r.projectionEvents)-1].Sales
}
