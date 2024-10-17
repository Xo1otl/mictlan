package stock

type KafkaProducer struct{}

// onAdded implements EventProducer.
func (k *KafkaProducer) onAdded(event AddedEvent) {
	panic("unimplemented")
}

// onAggregateUpdated implements EventProducer.
func (k *KafkaProducer) onAggregateUpdated(event AggregateUpdatedEvent) {
	panic("unimplemented")
}

// onClearedAll implements EventProducer.
func (k *KafkaProducer) onClearedAll(event ClearedAllEvent) {
	panic("unimplemented")
}

// onSold implements EventProducer.
func (k *KafkaProducer) onSold(event SoldEvent) {
	panic("unimplemented")
}

func NewKafkaProducer() EventProducer {
	return &KafkaProducer{}
}
