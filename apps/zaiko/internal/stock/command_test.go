package stock

import (
	"reflect"
	"testing"
	"zaiko/internal/auth"

	"github.com/shopspring/decimal"
)

func TestAggregate(t *testing.T) {
	var sub auth.Sub = "test-sub"
	agg := NewAggregate(nil)

	// ただのAdd
	addEvent, err := agg.Add(sub, "item1", 10)
	if err != nil {
		t.Errorf("Add returned error: %v", err)
	}
	expectedAddEvent := NewAddedEvent(sub, "item1", 10)
	if !reflect.DeepEqual(addEvent, expectedAddEvent) {
		t.Errorf("Add event mismatch: got %v, want %v", addEvent, expectedAddEvent)
	}
	if agg.stocks["item1"] != 10 {
		t.Errorf("Stock for item1 incorrect: got %d, want %d", agg.stocks["item1"], 10)
	}

	// 不正な追加
	_, err = agg.Add(sub, "item2", -5)
	if err == nil {
		t.Errorf("Expected error when adding negative amount, but got none")
	}

	// Sellのテスト
	sellEvent, err := agg.Sell(sub, "item1", 5, decimal.NewFromFloat(2.0))
	if err != nil {
		t.Errorf("Sell returned error: %v", err)
	}
	expectedSellEvent := NewSoldEvent(sub, "item1", 5, decimal.NewFromFloat(2.0))
	if !reflect.DeepEqual(sellEvent, expectedSellEvent) {
		t.Errorf("Sell event mismatch: got %v, want %v", sellEvent, expectedSellEvent)
	}
	if agg.stocks["item1"] != 5 {
		t.Errorf("Stock for item1 incorrect after sell: got %d, want %d", agg.stocks["item1"], 5)
	}
	if !agg.sales.Equal(decimal.NewFromFloat(10.0)) {
		t.Errorf("Sales incorrect: got %s, want %s", agg.sales.String(), decimal.NewFromFloat(10.0).String())
	}

	// 売りすぎ
	_, err = agg.Sell(sub, "item1", 10, decimal.NewFromFloat(2.0))
	if err == nil {
		t.Errorf("Expected error when selling more than stock, but got none")
	}

	// 0値でのSell
	_, err = agg.Sell(sub, "item1", 0, decimal.NewFromFloat(2.0))
	if err == nil {
		t.Errorf("Expected error when selling zero amount, but got none")
	}
	if agg.stocks["item1"] != 5 {
		t.Errorf("Stock for item1 incorrect after attempting to sell zero amount: got %d, want %d", agg.stocks["item1"], 5)
	}

	// ClearAllのテスト
	agg.ClearAll(sub)
	if len(agg.stocks) != 0 {
		t.Errorf("Stocks not cleared after ClearAll")
	}
	if agg.sales != decimal.Zero {
		t.Errorf("Sales not reset after ClearAll")
	}

	events := []any{
		NewAddedEvent(sub, "item3", 20),
		NewSoldEvent(sub, "item3", 5, decimal.NewFromFloat(3.0)),
	}
	agg2 := NewAggregate(events)
	if agg2.stocks["item3"] != 15 {
		t.Errorf("Stock for item3 incorrect in agg2: got %d, want %d", agg2.stocks["item3"], 15)
	}
	if !agg2.sales.Equal(decimal.NewFromFloat(15.0)) {
		t.Errorf("Sales incorrect in agg2: got %s, want %s", agg2.sales.String(), decimal.NewFromFloat(15.0).String())
	}
}

func TestCommand(t *testing.T) {
	var sub auth.Sub = "test-sub"
	eventStore := &InMemoryEventStore{}
	cmd := &Command{consumer: eventStore, producer: eventStore}

	// Add のテスト
	err := cmd.Add(sub, "item1", 10)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	expectedAddedEvent := NewAddedEvent(sub, "item1", 10)
	if len(eventStore.events) != 1 {
		t.Fatalf("expected 1 events, got %d", len(eventStore.events))
	}
	if !reflect.DeepEqual(eventStore.events[0], expectedAddedEvent) {
		t.Errorf("expected %v, got %v", expectedAddedEvent, eventStore.events[0])
	}

	expectedAggregateUpdated1 := NewAggregateUpdatedEvent(sub, map[string]int{"item1": 10}, decimal.Zero)
	if !reflect.DeepEqual(eventStore.projectionEvents[0], expectedAggregateUpdated1) {
		t.Errorf("expected %v, got %v", expectedAggregateUpdated1, eventStore.projectionEvents[0])
	}

	// Sell のテスト
	err = cmd.Sell(sub, "item1", 5, decimal.NewFromInt(100))
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	expectedSoldEvent := NewSoldEvent(sub, "item1", 5, decimal.NewFromInt(100))
	if len(eventStore.events) != 2 {
		t.Fatalf("expected 2 events, got %d", len(eventStore.events))
	}
	if !reflect.DeepEqual(eventStore.events[1], expectedSoldEvent) {
		t.Errorf("expected %v, got %v", expectedSoldEvent, eventStore.events[1])
	}

	expectedAggregateUpdated2 := NewAggregateUpdatedEvent(sub, map[string]int{"item1": 5}, decimal.NewFromInt(500))
	if !reflect.DeepEqual(eventStore.projectionEvents[1], expectedAggregateUpdated2) {
		t.Errorf("expected %v, got %v", expectedAggregateUpdated2, eventStore.projectionEvents[1])
	}

	// ClearAll のテスト
	cmd.ClearAll(sub)
	expectedClearedAllEvent := NewClearedAllEvent(sub)
	if len(eventStore.events) != 3 {
		t.Fatalf("expected 3 events, got %d", len(eventStore.events))
	}
	if !reflect.DeepEqual(eventStore.events[2], expectedClearedAllEvent) {
		t.Errorf("expected %v, got %v", expectedClearedAllEvent, eventStore.events[2])
	}

	expectedAggregateUpdated3 := NewAggregateUpdatedEvent(sub, map[string]int{}, decimal.Zero)
	if !reflect.DeepEqual(eventStore.projectionEvents[2], expectedAggregateUpdated3) {
		t.Errorf("expected %v, got %v", expectedAggregateUpdated3, eventStore.projectionEvents[2])
	}
}
