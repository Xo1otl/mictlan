package stock

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"

	"github.com/hamba/avro/v2"
	"github.com/twmb/franz-go/pkg/kgo"
	"github.com/twmb/franz-go/pkg/sr"
)

type KafkaProducer struct {
	client *kgo.Client

	keySchemaIDs   map[string]int         // Map topic to key schema ID
	keySchemas     map[string]avro.Schema // Map topic to key schema
	valueSchemaIDs map[string]int         // Map event type to value schema ID
	valueSchemas   map[string]avro.Schema // Map event type to value schema
	topics         map[string]string      // Map event type to topic
}

// NewKafkaProducer initializes a KafkaProducer with necessary schemas and clients.
func NewKafkaProducer(registryURL string, kafkaURL string) (EventProducer, error) {
	// Create Schema Registry client
	rcl, err := sr.NewClient(sr.URLs(registryURL))
	if err != nil {
		return nil, err
	}

	// Create Kafka client
	client, err := kgo.NewClient(
		kgo.SeedBrokers(kafkaURL),
	)
	if err != nil {
		return nil, err
	}

	producer := &KafkaProducer{
		client:         client,
		keySchemaIDs:   make(map[string]int),
		keySchemas:     make(map[string]avro.Schema),
		valueSchemaIDs: make(map[string]int),
		valueSchemas:   make(map[string]avro.Schema),
		topics:         make(map[string]string),
	}

	// Event types and their topics
	eventTopics := map[string]string{
		"Added":            "zaiko.stock.commands",
		"Sold":             "zaiko.stock.commands",
		"ClearedAll":       "zaiko.stock.commands",
		"AggregateUpdated": "zaiko.stock.projections",
	}

	for eventType, topic := range eventTopics {
		// Fetch key schema for the topic if not already fetched
		if _, exists := producer.keySchemas[topic]; !exists {
			keySchemaName := topic + "-key"
			keySchemaText, err := rcl.SchemaByVersion(context.Background(), keySchemaName, 1)
			if err != nil {
				return nil, err
			}
			keySchema, err := avro.Parse(keySchemaText.Schema.Schema)
			if err != nil {
				return nil, err
			}
			producer.keySchemaIDs[topic] = keySchemaText.ID
			producer.keySchemas[topic] = keySchema
		}

		// Fetch value schema for the event type
		valueSchemaName := topic + "-" + eventType + "-value"
		valueSchemaText, err := rcl.SchemaByVersion(context.Background(), valueSchemaName, 1)
		if err != nil {
			return nil, err
		}
		valueSchema, err := avro.Parse(valueSchemaText.Schema.Schema)
		if err != nil {
			return nil, err
		}
		producer.valueSchemaIDs[eventType] = valueSchemaText.ID
		producer.valueSchemas[eventType] = valueSchema

		// Store the topic for the event type
		producer.topics[eventType] = topic
	}

	return producer, nil
}

// produce is a helper method to serialize data and produce messages to Kafka.
func (k *KafkaProducer) produce(eventType string, keyData interface{}, valueData interface{}) error {
	ctx := context.Background()

	// Get the topic
	topic, ok := k.topics[eventType]
	if !ok {
		return fmt.Errorf("unknown event type: %s", eventType)
	}

	// Get key schema and ID
	keySchema, ok := k.keySchemas[topic]
	if !ok {
		return fmt.Errorf("key schema not found for topic: %s", topic)
	}
	keySchemaID := k.keySchemaIDs[topic]

	// Get value schema and ID
	valueSchema, ok := k.valueSchemas[eventType]
	if !ok {
		return fmt.Errorf("value schema not found for event type: %s", eventType)
	}
	valueSchemaID := k.valueSchemaIDs[eventType]

	// Serialize key
	keyBytes, err := avro.Marshal(keySchema, keyData)
	if err != nil {
		return err
	}

	// Serialize value
	valueBytes, err := avro.Marshal(valueSchema, valueData)
	if err != nil {
		return err
	}

	// Encode key and value
	keyEncoded := EncodeAvro(keySchemaID, keyBytes)
	valueEncoded := EncodeAvro(valueSchemaID, valueBytes)

	// Create record
	record := &kgo.Record{
		Topic: topic,
		Key:   keyEncoded,
		Value: valueEncoded,
	}

	// Produce message
	err = k.client.ProduceSync(ctx, record).FirstErr()
	if err != nil {
		return err
	}

	return nil
}

// OnAdded implements EventProducer.
func (k *KafkaProducer) OnAdded(event AddedEvent) error {
	keyData := event.Sub
	valueData := map[string]interface{}{
		"Sub":    event.Sub,
		"Name":   event.Name,
		"Amount": event.Amount,
	}

	return k.produce("Added", keyData, valueData)
}

// OnSold implements EventProducer.
func (k *KafkaProducer) OnSold(event SoldEvent) error {
	keyData := event.Sub
	valueData := map[string]interface{}{
		"Sub":    event.Sub,
		"Name":   event.Name,
		"Amount": event.Amount,
		"Price":  event.Price,
	}

	return k.produce("Sold", keyData, valueData)
}

// OnClearedAll implements EventProducer.
func (k *KafkaProducer) OnClearedAll(event ClearedAllEvent) error {
	keyData := event.Sub
	valueData := map[string]interface{}{
		"Sub": event.Sub,
	}

	return k.produce("ClearedAll", keyData, valueData)
}

// OnAggregateUpdated implements EventProducer.
func (k *KafkaProducer) OnAggregateUpdated(event AggregateUpdatedEvent) error {
	keyData := event.Sub
	valueData := map[string]interface{}{
		"Sub":    event.Sub,
		"Stocks": event.Stocks,
		"Sales":  event.Sales.String(),
	}
	log.Printf("%v", valueData)

	return k.produce("AggregateUpdated", keyData, valueData)
}

// EncodeAvro encodes the data with the magic byte and schema ID.
func EncodeAvro(schemaID int, data []byte) []byte {
	// Magic byte (0x00) + Schema ID (4 bytes) + serialized data
	payload := make([]byte, 1+4+len(data))
	payload[0] = 0
	binary.BigEndian.PutUint32(payload[1:5], uint32(schemaID))
	copy(payload[5:], data)
	return payload
}
