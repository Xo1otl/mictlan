package stock

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"zaiko/internal/auth"

	"github.com/hamba/avro/v2"
	"github.com/shopspring/decimal"
	"github.com/twmb/franz-go/pkg/kgo"
	"github.com/twmb/franz-go/pkg/sr"
)

type SchemaCache struct {
	cache map[int]avro.Schema
	rcl   *sr.Client
	mu    sync.Mutex // ゼロ値が有効なので初期化不要
}

func NewSchemaCache(rcl *sr.Client) *SchemaCache {
	return &SchemaCache{
		cache: make(map[int]avro.Schema),
		rcl:   rcl,
	}
}

func (sc *SchemaCache) GetSchema(schemaID int) (avro.Schema, error) {
	sc.mu.Lock()
	defer sc.mu.Unlock()

	// 既にキャッシュにある場合
	if schema, exists := sc.cache[schemaID]; exists {
		return schema, nil
	}

	// キャッシュになければスキーマレジストリから取得
	schemaText, err := sc.rcl.SchemaByID(context.Background(), schemaID)
	if err != nil {
		return nil, err
	}

	schema, err := avro.Parse(schemaText.Schema)
	if err != nil {
		return nil, err
	}

	// キャッシュに保存
	sc.cache[schemaID] = schema
	return schema, nil
}

type KafkaConsumer struct {
	client  *kgo.Client
	cache   *SchemaCache
	fetches kgo.Fetches
}

func NewKafkaConsumer(registryURL string, kafkaURL string) EventConsumer {
	// Create Schema Registry client
	rcl, err := sr.NewClient(sr.URLs(registryURL))
	if err != nil {
		log.Fatal(err)
	}
	// Create Kafka client
	client, err := kgo.NewClient(
		kgo.SeedBrokers(kafkaURL),
		kgo.ConsumeTopics("zaiko.stock.commands"),
	)
	if err != nil {
		log.Fatal(err)
	}
	sc := NewSchemaCache(rcl)
	consumer := &KafkaConsumer{cache: sc, client: client}
	// PollFetchesをバックグラウンドで実行
	go consumer.pollFetchesLoop()
	return consumer
}

func (k *KafkaConsumer) pollFetchesLoop() {
	ctx := context.Background()
	for {
		k.fetches = k.client.PollFetches(ctx)
		if k.fetches.IsClientClosed() {
			return
		}
		k.fetches.EachError(func(_ string, _ int32, err error) {
			panic(err)
		})
	}
}

func (k *KafkaConsumer) Events(sub auth.Sub) ([]any, error) {
	var results []any
	var lastError error

	k.fetches.EachRecord(func(record *kgo.Record) {
		var (
			keyData   auth.Sub
			valueData map[string]interface{}
		)

		// Process key
		if len(record.Key) < 5 {
			log.Panic("key too short")
			return
		}

		// Check magic byte
		if record.Key[0] != 0 {
			lastError = fmt.Errorf("unexpected magic byte in key")
			return
		}
		schemaID := int(binary.BigEndian.Uint32(record.Key[1:5]))
		schema, err := k.cache.GetSchema(schemaID)
		if err != nil {
			lastError = fmt.Errorf("failed to get key schema: %v", err)
			return
		}
		// Unmarshal key
		err = avro.Unmarshal(schema, record.Key[5:], &keyData)
		if err != nil {
			lastError = fmt.Errorf("failed to unmarshal key: %v", err)
			return
		}

		if keyData != sub {
			return
		}

		// Process value
		if len(record.Value) < 5 {
			log.Fatal("value too short")
		}

		// Check magic byte
		if record.Value[0] != 0 {
			lastError = fmt.Errorf("unexpected magic byte in value")
			return
		}
		schemaID = int(binary.BigEndian.Uint32(record.Value[1:5]))
		schema, err = k.cache.GetSchema(schemaID)
		if err != nil {
			lastError = fmt.Errorf("failed to get value schema: %v", err)
			return
		}
		// Unmarshal value
		err = avro.Unmarshal(schema, record.Value[5:], &valueData)
		if err != nil {
			lastError = fmt.Errorf("failed to unmarshal value: %v", err)
			return
		}

		type Schema struct {
			Name string `json:"name"`
		}
		var parsedSchema Schema
		err = json.Unmarshal([]byte(schema.String()), &parsedSchema)
		if err != nil {
			fmt.Println("Error:", err)
			return
		}
		var message any
		switch parsedSchema.Name {
		case "Added":
			message = NewAddedEvent(keyData, valueData["Name"].(string), valueData["Amount"].(int))
		case "Sold":
			price, err := decimal.NewFromString(valueData["Price"].(string))
			if err != nil {
				log.Fatal(err)
			}
			message = NewSoldEvent(keyData, valueData["Name"].(string), valueData["Amount"].(int), price)
		case "ClearedAll":
			message = NewClearedAllEvent(keyData)
		default:
			panic("Unknown schema")
		}
		results = append(results, message)
	})

	if lastError != nil {
		return nil, lastError
	}

	return results, nil
}
