package stock

import "github.com/shopspring/decimal"

// Commandのほうで使うEventConsumer/Producerの実装
type InMemoryEventStore struct {
	events           []any
	projectionEvents []AggregateUpdatedEvent
}

func (s *InMemoryEventStore) Events() []any {
	return s.events
}

func (s *InMemoryEventStore) onAdded(event AddedEvent) {
	s.events = append(s.events, event)
}

func (s *InMemoryEventStore) onSold(event SoldEvent) {
	s.events = append(s.events, event)
}

func (s *InMemoryEventStore) onClearedAll(event ClearedAllEvent) {
	s.events = append(s.events, event)
}

func (s *InMemoryEventStore) onAggregateUpdated(event AggregateUpdatedEvent) {
	s.projectionEvents = append(s.projectionEvents, event)
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
