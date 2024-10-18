package stock

import (
	"context"

	"github.com/twmb/franz-go/pkg/kgo"
	"github.com/twmb/franz-go/pkg/sr"

	"github.com/linkedin/goavro/v2"
)

type KafkaProducer struct {
	client   *kgo.Client
	srClient *sr.Client
	codecs   map[string]*goavro.Codec
}

func NewKafkaProducer() (EventProducer, error) {
	srClient, err := sr.NewClient(sr.URLs("http://redpanda:8081"))
	if err != nil {
		return nil, err
	}

	client, err := kgo.NewClient(
		kgo.SeedBrokers("redpanda:9092"),
	)
	if err != nil {
		return nil, err
	}

	return &KafkaProducer{
		client:   client,
		srClient: srClient,
		codecs:   make(map[string]*goavro.Codec),
	}, nil
}

func (k *KafkaProducer) getCodec(subject string) (*goavro.Codec, int32, error) {
	if codec, ok := k.codecs[subject]; ok {
		return codec, 0, nil
	}
	schemaResp, err := k.srClient.LookupSchema(context.Background(), subject, sr.Schema{})
	if err != nil {
		return nil, 0, err
	}

	codec, err := goavro.NewCodec(schemaResp.Schema.Type.String())
	if err != nil {
		return nil, 0, err
	}

	k.codecs[subject] = codec

	return codec, int32(schemaResp.ID), nil
}

func (k *KafkaProducer) encodeMessage(binary []byte, schemaID int32) ([]byte, error) {
	message := make([]byte, 0, 5+len(binary))
	message = append(message, 0) // マジックバイト
	message = append(message, byte(schemaID>>24), byte(schemaID>>16), byte(schemaID>>8), byte(schemaID))
	message = append(message, binary...)
	return message, nil
}

func (k *KafkaProducer) produceMessage(topic string, key interface{}, value interface{}, keySubject string, valueSubject string) error {
	keyCodec, keySchemaID, err := k.getCodec(keySubject)
	if err != nil {
		return err
	}

	keyBinary, err := keyCodec.BinaryFromNative(nil, key)
	if err != nil {
		return err
	}

	valueCodec, valueSchemaID, err := k.getCodec(valueSubject)
	if err != nil {
		return err
	}

	valueBinaryNative, err := valueCodec.BinaryFromNative(nil, value)
	if err != nil {
		return err
	}

	valueMessage, err := k.encodeMessage(valueBinaryNative, valueSchemaID)
	if err != nil {
		return err
	}

	keyMessage, err := k.encodeMessage(keyBinary, keySchemaID)
	if err != nil {
		return err
	}

	k.client.Produce(context.Background(), &kgo.Record{
		Topic: topic,
		Key:   keyMessage,
		Value: valueMessage,
	}, nil)

	return nil
}

func (k *KafkaProducer) onAdded(event AddedEvent) error {
	keySubject := "zaiko.stock.commands-key"
	valueSubject := "zaiko.stock.commands-Added-value"
	topic := "zaiko.stock.commands"

	value := map[string]interface{}{
		"Sub":    event.Sub,
		"Name":   event.Name,
		"Amount": event.Amount,
	}

	return k.produceMessage(topic, event.Sub, value, keySubject, valueSubject)
}

func (k *KafkaProducer) onSold(event SoldEvent) error {
	keySubject := "zaiko.stock.commands-key"
	valueSubject := "zaiko.stock.commands-Sold-value"
	topic := "zaiko.stock.commands"

	value := map[string]interface{}{
		"Sub":    event.Sub,
		"Name":   event.Name,
		"Amount": event.Amount,
		"Price":  event.Price.String(),
	}

	return k.produceMessage(topic, event.Sub, value, keySubject, valueSubject)
}

func (k *KafkaProducer) onClearedAll(event ClearedAllEvent) error {
	keySubject := "zaiko.stock.commands-key"
	valueSubject := "zaiko.stock.commands-ClearedAll-value"
	topic := "zaiko.stock.commands"

	value := map[string]interface{}{
		"Sub": event.Sub,
	}

	return k.produceMessage(topic, event.Sub, value, keySubject, valueSubject)
}

func (k *KafkaProducer) onAggregateUpdated(event AggregateUpdatedEvent) error {
	keySubject := "zaiko.stock.projections-key"
	valueSubject := "zaiko.stock.projections-AggregateUpdated-value"
	topic := "zaiko.stock.projections"

	stocks := make(map[string]interface{})
	for k, v := range event.Stocks {
		stocks[k] = v
	}

	value := map[string]interface{}{
		"Stocks": stocks,
		"Sales":  event.Sales.String(),
	}

	return k.produceMessage(topic, event.Sub, value, keySubject, valueSubject)
}
