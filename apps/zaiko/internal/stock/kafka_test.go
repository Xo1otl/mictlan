package stock_test

import (
	"context"
	"fmt"
	"hash/crc32"
	"log"
	"testing"
	"zaiko/internal/stock"

	"github.com/hamba/avro/v2"
	"github.com/twmb/franz-go/pkg/kgo"
	"github.com/twmb/franz-go/pkg/kmsg"
	"github.com/twmb/franz-go/pkg/sr"
)

var (
	// KafkaとSchema RegistryのURL
	registry = "redpanda:8081"
	kafkaURL = "redpanda:9092"
	topic    = "zaiko.stock.projections"
	groupID  = "test-group"
)

func TestProduce(t *testing.T) {
	// Schema Registryクライアントの作成
	rcl, err := sr.NewClient(sr.URLs(registry))
	if err != nil {
		t.Fatal(err)
	}

	// スキーマの取得 (キーとバリューのスキーマ)
	keySchemaText, err := rcl.SchemaByVersion(context.Background(), "zaiko.stock.projections-key", 1)
	if err != nil {
		t.Fatal(err)
	}
	valueSchemaText, err := rcl.SchemaByVersion(context.Background(), "zaiko.stock.projections-AggregateUpdated-value", 1)
	if err != nil {
		t.Fatal(err)
	}

	// スキーマのパース
	keySchema, err := avro.Parse(keySchemaText.Schema.Schema)
	if err != nil {
		t.Fatal(err)
	}
	valueSchema, err := avro.Parse(valueSchemaText.Schema.Schema)
	if err != nil {
		t.Fatal(err)
	}

	// データの準備
	keyData := "testusersub"
	valueData := map[string]interface{}{
		"Stocks": map[string]int{
			"item1": 100,
			"item2": 200,
		},
		"Sales": "480.0",
	}

	// キーとバリューのシリアライズ
	keyBytes, err := avro.Marshal(keySchema, keyData)
	if err != nil {
		t.Fatal(err)
	}
	valueBytes, err := avro.Marshal(valueSchema, valueData)
	if err != nil {
		t.Fatal(err)
	}

	// マジックバイトとスキーマIDを付加
	keyEncoded := stock.EncodeAvro(keySchemaText.ID, keyBytes)
	valueEncoded := stock.EncodeAvro(valueSchemaText.ID, valueBytes)

	// Kafkaプロデューサーの作成
	client, err := kgo.NewClient(
		kgo.SeedBrokers(kafkaURL),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	// メッセージの送信
	record := &kgo.Record{
		Topic: topic,
		Key:   keyEncoded,
		Value: valueEncoded,
	}
	err = client.ProduceSync(context.Background(), record).FirstErr()
	if err != nil {
		t.Fatal(err)
	}

	t.Log("メッセージのプロデュースに成功しました")
}

func TestConsume(t *testing.T) {
	// Schema Registryクライアントの作成
	rcl, err := sr.NewClient(sr.URLs(registry))
	if err != nil {
		t.Fatal(err)
	}

	// スキーマの取得 (キーとバリューのスキーマ)
	keySchemaText, err := rcl.SchemaTextByVersion(context.Background(), "zaiko.stock.projections-key", 1)
	if err != nil {
		t.Fatal(err)
	}
	valueSchemaText, err := rcl.SchemaTextByVersion(context.Background(), "zaiko.stock.projections-AggregateUpdated-value", 1)
	if err != nil {
		t.Fatal(err)
	}

	// スキーマのパース
	keySchema, err := avro.Parse(keySchemaText)
	if err != nil {
		t.Fatal(err)
	}
	valueSchema, err := avro.Parse(valueSchemaText)
	if err != nil {
		t.Fatal(err)
	}

	// Kafkaコンシューマーの作成
	client, err := kgo.NewClient(
		kgo.SeedBrokers(kafkaURL),
		kgo.ConsumerGroup(groupID),
		kgo.ConsumeTopics(topic),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	// メッセージの受信
	ctx := context.Background()
	fetches := client.PollFetches(ctx)
	errs := fetches.Errors()
	if len(errs) > 0 {
		for _, err := range errs {
			t.Error(err)
		}
	}

	// メッセージの処理
	fetches.EachRecord(func(record *kgo.Record) {
		// キーのデシリアライズ
		var key string
		err := avro.Unmarshal(keySchema, record.Key[5:], &key)
		if err != nil {
			t.Error(err)
		}

		// バリューのデシリアライズ
		var value map[string]interface{}
		err = avro.Unmarshal(valueSchema, record.Value[5:], &value)
		if err != nil {
			t.Error(err)
		}

		t.Logf("メッセージをコンシュームしました: key=%v, value=%v", key, value)
	})
}

func TestCreateSchema(t *testing.T) {
	rcl, err := sr.NewClient(sr.URLs(registry))
	if err != nil {
		t.Fatal(err)
	}

	// スキーマとサブジェクトの定義
	var schemas = map[string]string{
		"zaiko.stock.commands-Added-value": `{
			"type": "record",
			"name": "Added",
			"fields": [
				{"name": "Sub", "type": "string"},
				{"name": "Name", "type": "string"},
				{"name": "Amount", "type": "int"}
			]
		}`,
		"zaiko.stock.commands-ClearedAll-value": `{
			"type": "record",
			"name": "ClearedAll",
			"fields": [
				{"name": "Sub", "type": "string"}
			]
		}`,
		"zaiko.stock.commands-Sold-value": `{
			"type": "record",
			"name": "Sold",
			"fields": [
				{"name": "Sub", "type": "string"},
				{"name": "Name", "type": "string"},
				{"name": "Amount", "type": "int"},
				{"name": "Price", "type": "string"}
			]
		}`,
		"zaiko.stock.commands-key": `{
			"name": "sub",
			"type": "string"
		}`,
		"zaiko.stock.projections-AggregateUpdated-value": `{
			"type": "record",
			"name": "AggregateUpdated",
			"fields": [
				{"name": "Sub", "type": "string"},
				{
					"name": "Stocks",
					"type": {
						"type": "map",
						"values": "int"
					}
				},
				{"name": "Sales", "type": "string"}
			]
		}`,
		"zaiko.stock.projections-key": `{
			"name": "Sub",
			"type": "string"
		}`,
	}

	// スキーマ作成処理のループ
	for subject, schemaText := range schemas {
		ss, err := rcl.CreateSchema(context.Background(), subject, sr.Schema{
			Schema: schemaText,
			Type:   sr.TypeAvro,
		})
		if err != nil {
			t.Fatalf("Failed to create schema for subject %s: %v", subject, err)
		}
		log.Printf("Created schema with ID: %d for subject: %s", ss.ID, subject)
	}
}

func TestKafkaProducer(t *testing.T) {
	producer, err := stock.NewKafkaProducer("redpanda:8081", "redpanda:9092")
	if err != nil {
		t.Fatal(err)
	}
	producer.OnClearedAll(stock.NewClearedAllEvent("testusersub"))
	producer.OnAdded(stock.NewAddedEvent("testusersub", "item1", 100))
}

func TestKafkaConsumer(t *testing.T) {
	consumer := stock.NewKafkaConsumer("redpanda:8081", "redpanda:9092")
	events, err := consumer.Events("testusersub")
	if err != nil {
		t.Fatal(err)
	}
	t.Log(events)
}

// Kafkaのデフォルトのハッシュ関数（Murmur2）に基づくカスタムハッシュ関数
func hashKey(key []byte) int32 {
	crc32q := crc32.MakeTable(0xedb88320)
	return int32(crc32.Checksum(key, crc32q))
}

func TestFindKey(t *testing.T) {
	// Kafkaクライアントの作成
	cl, err := kgo.NewClient(
		kgo.SeedBrokers("redpanda:9092"), // ブローカーのアドレス
	)
	if err != nil {
		panic(err)
	}
	defer cl.Close()

	// トピック名と取得したいキー
	topic := "zaiko.stock.projections"
	key := []byte("testusersub")

	// トピックのメタデータを取得
	metaRequest := kmsg.NewMetadataRequest()
	metaRequestTopic := kmsg.NewMetadataRequestTopic()
	metaRequestTopic.Topic = kmsg.StringPtr(topic)
	metaRequest.Topics = append(metaRequest.Topics, metaRequestTopic)

	metaResponse, err := cl.Request(context.Background(), &metaRequest)
	if err != nil {
		panic(err)
	}

	// トピックのパーティション数を取得
	var partitionCount int32
	for _, topicMetadata := range metaResponse.(*kmsg.MetadataResponse).Topics {
		if *topicMetadata.Topic == topic {
			partitionCount = int32(len(topicMetadata.Partitions))
			break
		}
	}

	if partitionCount == 0 {
		fmt.Println("トピックが見つかりませんでした。")
		return
	}

	// key のハッシュ値からパーティションを計算
	partition := hashKey(key) % partitionCount
	if partition < 0 {
		partition = -partition
	}

	fmt.Printf("Key '%s' はパーティション %d にあります。\n", key, partition)
}
