package stock

import (
	"fmt"
	"zaiko/internal/auth"

	"github.com/shopspring/decimal"
)

// projectionをいちいちつくるの大変だしaggregateがprojectionしたいデータ持ってるので
// ここでprojection用のeventも作ってproduce
type Command struct {
	consumer EventConsumer
	producer EventProducer
}

func NewCommand(consumer EventConsumer, producer EventProducer) *Command {
	return &Command{consumer: consumer, producer: producer}
}

func (c *Command) Add(sub auth.Sub, name string, amount int) error {
	events, err := c.consumer.Events(sub)
	if err != nil {
		return err
	}
	agg := NewAggregate(events)
	addedEvent, err := agg.Add(sub, name, amount)
	if err != nil {
		return err
	}
	err = c.producer.OnAdded(addedEvent)
	if err != nil {
		return err
	}
	// projectionもやる
	err = c.producer.OnAggregateUpdated(NewAggregateUpdatedEvent(sub, agg.stocks, agg.sales))
	if err != nil {
		return err
	}
	return nil
}

func (c *Command) Sell(sub auth.Sub, name string, amount int, price decimal.Decimal) error {
	events, err := c.consumer.Events(sub)
	if err != nil {
		return err
	}
	agg := NewAggregate(events)
	soldEvent, err := agg.Sell(sub, name, amount, price)
	if err != nil {
		return err
	}
	err = c.producer.OnSold(soldEvent)
	if err != nil {
		return err
	}
	err = c.producer.OnAggregateUpdated(NewAggregateUpdatedEvent(sub, agg.stocks, agg.sales))
	if err != nil {
		return err
	}
	return nil
}

func (c *Command) ClearAll(sub auth.Sub) error {
	events, err := c.consumer.Events(sub)
	if err != nil {
		return err
	}
	agg := NewAggregate(events)
	clearedAllEvent := agg.ClearAll(sub)
	err = c.producer.OnClearedAll(clearedAllEvent)
	if err != nil {
		return err
	}
	err = c.producer.OnAggregateUpdated(NewAggregateUpdatedEvent(sub, agg.stocks, agg.sales))
	if err != nil {
		return err
	}
	return nil
}

type EventConsumer interface {
	Events(sub auth.Sub) ([]any, error)
}

type EventProducer interface {
	OnAdded(event AddedEvent) error
	OnSold(event SoldEvent) error
	OnClearedAll(event ClearedAllEvent) error
	OnAggregateUpdated(event AggregateUpdatedEvent) error
}

type AddedEvent struct {
	Sub    auth.Sub
	Name   string
	Amount int
}

func NewAddedEvent(sub auth.Sub, name string, amount int) AddedEvent {
	return AddedEvent{Sub: sub, Name: name, Amount: amount}
}

type SoldEvent struct {
	Sub    auth.Sub
	Name   string
	Amount int
	Price  decimal.Decimal
}

func NewSoldEvent(sub auth.Sub, name string, amount int, price decimal.Decimal) SoldEvent {
	return SoldEvent{Sub: sub, Name: name, Amount: amount, Price: price}
}

type ClearedAllEvent struct {
	Sub auth.Sub
}

func NewClearedAllEvent(sub auth.Sub) ClearedAllEvent {
	return ClearedAllEvent{Sub: sub}
}

type AggregateUpdatedEvent struct {
	Sub    auth.Sub
	Stocks map[string]int
	Sales  decimal.Decimal
}

func NewAggregateUpdatedEvent(sub auth.Sub, stocks map[string]int, sales decimal.Decimal) AggregateUpdatedEvent {
	return AggregateUpdatedEvent{Sub: sub, Stocks: stocks, Sales: sales}
}

type Aggregate struct {
	Sub    auth.Sub
	stocks map[string]int
	sales  decimal.Decimal
}

func NewAggregate(events []any) *Aggregate {
	agg := &Aggregate{
		stocks: make(map[string]int),
		sales:  decimal.Zero,
	}
	for _, event := range events {
		agg.Apply(event)
	}
	return agg
}

func (a *Aggregate) Add(sub auth.Sub, name string, amount int) (AddedEvent, error) {
	if amount <= 0 {
		return AddedEvent{}, fmt.Errorf("amountが不正です")
	}
	event := NewAddedEvent(sub, name, amount)
	a.Apply(event)
	return event, nil
}

func (a *Aggregate) Sell(sub auth.Sub, name string, amount int, price decimal.Decimal) (SoldEvent, error) {
	// priceがゼロはありえなくはないが、数がゼロなのは不正
	if amount <= 0 {
		return SoldEvent{}, fmt.Errorf("amountが不正です")
	}
	// priceの小数点以下の桁数をチェック（例: 2桁まで許可）
	if price.Exponent() < -8 {
		return SoldEvent{}, fmt.Errorf("priceが不正です: 小数点以下は8桁までです")
	}
	currentStock := a.stocks[name]
	if currentStock < amount {
		return SoldEvent{}, fmt.Errorf("stockが足りません")
	}
	event := NewSoldEvent(sub, name, amount, price)
	a.Apply(event)
	return event, nil
}

func (a *Aggregate) ClearAll(sub auth.Sub) ClearedAllEvent {
	event := NewClearedAllEvent(sub)
	a.Apply(event)
	return event
}

func (a *Aggregate) Apply(event any) {
	switch e := event.(type) {
	case AddedEvent:
		a.stocks[e.Name] += e.Amount
	case SoldEvent:
		a.stocks[e.Name] -= e.Amount
		a.sales = a.sales.Add(decimal.NewFromInt(int64(e.Amount)).Mul(e.Price))
	case ClearedAllEvent:
		a.stocks = make(map[string]int)
		a.sales = decimal.Zero
	}
}
