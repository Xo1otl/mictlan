package stock

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/twmb/franz-go/pkg/kadm"
	"github.com/twmb/franz-go/pkg/kgo"
	"github.com/twmb/franz-go/pkg/kversion"
	"github.com/twmb/franz-go/pkg/sr"
)

func CreateTopics() {
	seeds := []string{KafkaURL}

	var adminClient *kadm.Client
	{
		client, err := kgo.NewClient(
			kgo.SeedBrokers(seeds...),

			// Do not try to send requests newer than 2.4.0 to avoid breaking changes in the request struct.
			// Sometimes there are breaking changes for newer versions where more properties are required to set.
			kgo.MaxVersions(kversion.V2_4_0()),
		)
		if err != nil {
			panic(err)
		}
		defer client.Close()

		adminClient = kadm.NewClient(client)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	topicDetails, err := adminClient.ListTopics(ctx)
	if err != nil {
		log.Fatalf("failed to list topics: %v", err)
	}

	// Create topic "zaiko.stock.projections" if it doesn't exist already
	projectionsTopic := "zaiko.stock.projections"
	if topicDetails.Has(projectionsTopic) {
		fmt.Printf("topic %v already exists\n", projectionsTopic)
		return
	}
	fmt.Printf("Creating topic %v\n", projectionsTopic)

	createTopicResponse, err := adminClient.CreateTopic(ctx, -1, -1, nil, projectionsTopic)
	if err != nil {
		log.Fatalf("failed to create topic: %v", err)
	}
	fmt.Printf("Successfully created topic %v\n", createTopicResponse.Topic)

	// Create topic "zaiko.stock.commands" if it doesn't exist already
	commandsTopic := "zaiko.stock.commands"
	if topicDetails.Has(commandsTopic) {
		fmt.Printf("topic %v already exists\n", projectionsTopic)
		return
	}
	fmt.Printf("Creating topic %v\n", projectionsTopic)

	createTopicResponse, err = adminClient.CreateTopic(ctx, -1, -1, nil, commandsTopic)
	if err != nil {
		log.Fatalf("failed to create topic: %v", err)
	}
	fmt.Printf("Successfully created topic %v\n", createTopicResponse.Topic)
}

func RegisterSchema() {
	rcl, err := sr.NewClient(sr.URLs(Registry))
	if err != nil {
		log.Fatal(err)
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
			log.Fatalf("Failed to create schema for subject %s: %v", subject, err)
		}
		log.Printf("Created schema with ID: %d for subject: %s", ss.ID, subject)
	}
}

var (
	Registry = "redpanda:8081"
	KafkaURL = "redpanda:9092"
)
