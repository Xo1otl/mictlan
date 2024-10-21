## 課題1

- 課題2が終わってから`internal/apiserver/echo.go`にエンドポイントを追加した
```go
e.GET("/", func(c echo.Context) error {
	return c.String(200, "AWS")
})
```

## 課題2

### `auth` モジュールを作った

- バックエンドで必要な認証処理を行うミドルウェアを作成した。
    - 過去に JWT 認証で使用したエンティティ (`Token`, `Claims`) とインターフェース (`TokenService`) を `internal/auth/*` に持ってきた。
    - ミドルウェア `internal/auth/echomiddleware.go` をDigest認証に合わせて微修正した。

### `iam` モジュールを作った

- ドメインレイヤを作った
    - `internal/iam/digest.go` で nonce 発行 (`Init`) とトークンのパース (`Parse`) を行う `Digest` を作った。
    - インターフェースを定義した 
        - アカウント情報を取得する `AccountRepo`。
        - リプレイ攻撃防止のための nonce カウント (`Nc`) の管理/検証 `DigestNcRepo`。
        - nonce の生成と改ざん防止のための検証 `NonceService`。
        - Digest レスポンスの検証 `DigestTokenValidator`。
- インターフェースを実装した
    - MD5を用いたトークン検証 `MD5DigestValidator` (`internal/iam/validator.go`)
    - nonceカウント (`Nc`) のrepo `InMemoryDigestNcRepo` (`internal/iam/inmemory.go`)
    - HMACを用いた nonce の生成/検証 `HMACNonceService` (`internal/iam/nonceservice.go`)
    - アカウントのrepo `InMemoryAccountRepo` (`internal/iam/inmemory.go`)。
- Digest認証のチャレンジとレスポンスの流れを処理するミドルウェア `EchoDigestMiddleware` を実装した (`internal/iam/echodigestmiddleware.go`)。

### `apiserver` モジュールを作った

- `internal/apiserver/echo.go` を書いた
    - Echo サーバーを初期化。
    - CORS 設定の構成(今後のためにコード残してある)。
    - `iam` の各コンポーネント（アカウントリポジトリ、nonce サービス、バリデータなど）を依存性注入。
    - ミドルウェアチェーンに `iam.EchoDigestMiddleware` と `auth.EchoMiddleware` を追加。
    - ルート（例：`/secret` エンドポイント）の登録。
    - サーバーの起動。

### 反省点

- Nc等のリソース解放していない

### 参考文献

- [Digest認証の仕様](https://datatracker.ietf.org/doc/html/rfc7616)
    - `The nc value is the hexadecimal count of the number of requests (including the current request) that the client has sent with the noncevalue in this request. `
- [Digest認証日本語訳](https://tex2e.github.io/rfc-translater/html/rfc7616.html)
- [Digest認証日本語解説](https://kunishi.gitbook.io/web-application-textbook/storage)
- [RubyのDigest Client](https://www.rubydoc.info/gems/net-http-digest_auth/1.1.1/Net/HTTP/DigestAuth)

## 課題3

### `stock` モジュールを作った

- ドメイン設計を行った
    - `stock`モジュール内で「stock」がすべての主語となる命名にした。
        - 例: Event は `StockEvent`、Aggregate は `StockAggregate`、Add 関数は `AddStock` という形で、主語が省略されている。
        - Repository Pattern に倣い、具体的なプレフィックスがつく構造。例えば `UserRepository` と同じように考えているけど、主語が省略されている。
    - `producer` や `consumer` の設計も Repository Pattern を参考にした。ただし、`command service` としての処理のみを考えた。
    - `query service` は、projection された read model に対して repository pattern を用いてデータを取得する。

- projection の取り扱いを考えた
    - Projection を別サービスにすることも検討したが、command service が aggregate した結果を用いれば十分と判断した。
    - そのため、`command service` が projection の処理を兼任し、aggregate の結果を produce。
    - Kafka Connect を用いて、aggregate のトピックを consume し、read model に反映する予定だが、いったんmockを作って全部動作確認した。
    
- ドメインレイヤを書いた
    - `command.go`にDomainと必要なInterfaceを実装した。
    
- Mockサービスを実装した
    - Command Service の EventProducer と EventConsumer `InMemoryEventStore` (`inmemory.go`)
    - Query Service Repository `InMemoryRepo` (`inmemory.go`)

- price と sales の扱いを考え直した
    - もともと文字列だったが、480.0という数字にする必要があることに気づいた
    - 金額のような精度が求められる計算では浮動小数点による誤差を考える必要があることを思い出した
    - 高精度な計算が必要と判断し、decimal ライブラリを使用して金額計算を実装しなおした
    
- Kafkaを準備
    - dockerを使用してredpandaとredpanda consoleを立てた
    - 過去に趣味でかいたKafkaのdocker composeを再利用
    - zaiko.stockのzaoko.stock_projectionの二つのtopicとschemaを用意した
        - `rpk --brokers redpanda:9092 topic create zaiko.stock.commands`
        - `rpk --brokers redpanda:9092 topic create zaiko.stock.projections`
    - valueのstrategyはtopic&record, avroを使用
        - zaiko.stockは在庫管理における様々なイベントを持ち、すべて型が異なる
        - zaoko.stock_projectionは拡張性を考えて設計
    - keyのstrategyは、record毎に代わるわけではないと判断してtopicだけの方
    
- Kafkaを使った実装を行った
    - avro schema
        - 同じaggregateに対して複数の型のeventがあるため、strategyはtopic+recordにした
        - keyのstrategyはaggregate単位で考え、topicにした
    - KafkaClientのPoCコード
        - `kafka_test.go`にprojection用のeventをproduceするテスト関数と、consumeするテスト関数を用意した
        - magic byteがないとredpanda console上で表示できないことがわかり、修正した
    - `KafkaProducer` (`kafkaproducer.go`)
        - avroのschemaのkeyはsubにすべきなので、すべてのentityが正しくsubを持つよう修正、echorouteではMockの認証claimをセットするようにした
    - `KafkaConsumer` (`kafkaconsumer.go`)
        - kgo.PollFetchをgoroutineでループしており、リアルタイムにeventが更新される
        - mutexを使用し、eventの更新中はrecordsをロックしている
    - MockのProcessorのKafkaConnector `stockconnector.yaml`
        - yamlは手書きせずにpythonで生成
        - 最初はstdoutに出力するmock connectorを書いた
    - Query Serviceのデータベースとしてmongodbを採用した
        - Elasticsearch, Meilisearch等を調べてどれにするか迷った
        - 全文検索エンジンはメモリが足りないず不要と判断した
        - mysqlの場合も調べた
            - redpanda connectではprepared statementでmulti queryができない等の制約があった
            - stored procedureやsql viewやprepared statementについて詳しくなった
        - mongodbはredpanda connectを使えば直接exportできるし、documentの検索もできて便利だった
    - Mongodb用のKafkaConnectorのProcessorを書いた (`stockconnector.yaml`)
        - pythonで生成
        - eventには新しいsalesの結果とstocksが入っているので、それを反映する
    - Repository Adapterを実装 (`mongorepo.go`)
        - grafanaでmongodb見ようとしたが、enterprise liscenseが必要だった、community版も試したが機能がイマイチ
        - `mongodb.mongodb-vscode`が便利だった


### 反省点
    - stocksをmapにしているが、こういうのはmapのarrayとして扱う方が拡張性が高い気もする
    - kafkaだけではkeyによるフィルタリングもできないため不自然な処理になっている気もする
        - cassandraにexportしてconsumerはcassandraからイベントを取得するようにするか迷った
        - connector書いて、eventconsumerのadapterを用意するだけなので実装は現実的
        - event sourcingについて調べてもkafkaにproduceしてcassandraにexportする方法について実践的な記事や動画が見当たらないため慎重になった
        - kafka stream, ksqlDB, apache flink等の例があるが、java platformは大体重いので手軽に利用できない、Materializeはrust製のため使ってみたいが、cloud版のみ
        - partitionを増やしてkeyの存在するpartitionに対してのみクエリを行えば効率化でき、snapshottingも利用すればさらに効率化可能なためkafkaだけでもいいかもしれない
        - keyごとのpartitionというのが現実的なのかどうかわからない
    - redpandaもmysqlもproduction環境ではなく認証もない状態であること
        - helmを勉強してkubernetsクラスタにしたり、分散処理やnamespaceや認証などの設定が必要

### 確認ポイント

- APIとデータベースの関係性:
    - APIはデータベースと連携することでステートレスを実現できる
- HTTP メソッドと API の理解:
    - HTTP メソッド（GET、POST、DELETE など）の使い方と意味を理解。
    - curl コマンドの `-d` オプションで POST リクエストを送信できることを学習。
- API の実行と確認:
    - 実装した5つのAPIを、curl コマンドを用いて同じ結果が得られることを確認。
    - 異常系（不正な値のリクエストやエラー処理）にも対応するよう実装。
    - 価格の計算精度や変数ごとの 0 の扱いに注意し、関連しないメソッドに対してもエラーを出すように設定。

### 参考文献

- [avorについて](https://docs.oracle.com/cd/E35584_01/html/GettingStartedGuide/avroschemas.html)
- [プログラムの計算精度](https://zenn.dev/sdb_blog/articles/01_plus_02_does_not_equal_03)
- [decimalについて](https://engineering.mercari.com/blog/entry/20201203-basis-point/)
- [decimalパッケージ](https://github.com/shopspring/decimal)
- [Event Driven Architecture](https://aws.amazon.com/what-is/eda/)
- [Kafka Client](https://docs.redpanda.com/redpanda-labs/clients/docker-go/)
- [redpanda connect mysql](https://docs.redpanda.com/redpanda-connect/components/processors/sql_raw/?tab=tabs-2-table-insert-mysql)
- [event sourcing](https://youtube.com/playlist?list=PLa7VYi0yPIH1TXGUoSUqXgPMD2SQXEXxj)
- [ES and CQRS](https://youtu.be/MYD4rrIqDhA)
- [bloblang](https://docs.redpanda.com/redpanda-connect/guides/bloblang/methods/#key_values)
- [multi sql insert(not supported)](https://github.com/redpanda-data/connect/issues/1495)
- [cassandra db](https://www.scylladb.com/)
- [cassandra output connector](https://docs.redpanda.com/redpanda-connect/components/outputs/cassandra/)
- [influx db](https://github.com/influxdata/influxdb)
- [stored procedure](https://dev.mysql.com/doc/refman/8.0/ja/create-procedure.html)
- [docker compose amazon linux](https://qiita.com/mitarashi_cookie/items/92b03e1350cb80e64f64)
- [docker compose](https://github.com/docker/compose#linux)

### ソースコード
- `command.go`
```go
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
	// priceの小数点以下の桁数をチェック
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
```
- `query.go`
```go
package stock

import (
	"zaiko/internal/auth"

	"github.com/shopspring/decimal"
)

type Repo interface {
	Stocks(sub auth.Sub, name string) map[string]int
	Sales(sub auth.Sub) decimal.Decimal
}
```
- `avro schema`
```json
{
    "zaiko.stock.commands-Added-value": {
        "type": "record",
        "name": "Added",
        "fields": [
            {"name": "Sub", "type": "string"},
            {"name": "Name", "type": "string"},
            {"name": "Amount", "type": "int"}
        ]
    },
    "zaiko.stock.commands-ClearedAll-value": {
        "type": "record",
        "name": "ClearedAll",
        "fields": [
            {"name": "Sub", "type": "string"}
        ]
    },
    "zaiko.stock.commands-Sold-value": {
        "type": "record",
        "name": "Sold",
        "fields": [
            {"name": "Sub", "type": "string"},
            {"name": "Name", "type": "string"},
            {"name": "Amount", "type": "int"},
            {"name": "Price", "type": "string"}
        ]
    },
    "zaiko.stock.commands-key": {
        "name": "sub",
        "type": "string"
    },
    "zaiko.stock.projections-AggregateUpdated-value": {
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
    },
    "zaiko.stock.projections-key": {
        "name": "Sub",
        "type": "string"
    }
}
```
- `stockprojection.tpl.py`
```python
import yaml
import os
from infra import broker
from infra import zaiko
from infra import nosql

sales_stocks_mongo = {
    "label": "sales_stocks_mongo",
    "mongodb": {
        "url": f"mongodb://{nosql.CONTAINER_NAME}:{nosql.PORT}",
        "database": zaiko.DB_NAME,
        "collection": "sales_stocks",
        "operation": "replace-one",
        "write_concern": {
            "w": "majority",
            "j": True
        },
        "upsert": True,
        "document_map": """
root.Sub = this.Sub
root.Stocks = this.Stocks
root.Sales = this.Sales
""",
        "filter_map": "root.Sub = this.Sub"
    }
}

stock_connector = {
    "input": {
        "kafka_franz": {
            "seed_brokers": [broker.KAFKA_ADDR],
            "topics": ["zaiko.stock.projections"],
            "consumer_group": "zaiko.stock.projector",
            "auto_replay_nacks": True,
        }
    },
    "processor_resources": [
        sales_stocks_mongo
    ],
    "pipeline": {
        "processors": [
            {
                "schema_registry_decode": {
                    "url": broker.SCHEMA_REGISTRY_URL,
                }
            },
            {
                "try": [
                    {"resource": sales_stocks_mongo["label"]},
                ]
            },
            {
                "catch": [{
                    "log": {
                        "message": "Processing failed due to: ${!error()}"
                    }
                }]
            }
        ]
    },
    "output": {
        "stdout": {}
    }
}

target = os.path.join(os.path.dirname(__file__), "stockconnector.yaml")

with open(target, 'w') as file:
    yaml.dump(stock_connector, file)

print(f"[zaiko] stockconnector has been written to {target}.")
```
- `docker compose config redpanda redpanda-console mongo zaiko`
```
name: awsjob
services:
  mongo:
    container_name: mongo
    image: mongo:latest
    networks:
      default: null
    ports:
      - mode: ingress
        target: 27017
        published: "27017"
        protocol: tcp
  redpanda:
    command:
      - redpanda
      - start
      - --kafka-addr internal://0.0.0.0:9092,external://0.0.0.0:19092
      - --advertise-kafka-addr internal://redpanda:9092,external://localhost:19092
      - --pandaproxy-addr internal://0.0.0.0:8082,external://0.0.0.0:18082
      - --advertise-pandaproxy-addr internal://redpanda:8082,external://localhost:18082
      - --schema-registry-addr internal://0.0.0.0:8081,external://0.0.0.0:18081
      - --rpc-addr redpanda:33145
      - --advertise-rpc-addr redpanda:33145
      - --mode dev-container
      - --smp 1
      - --default-log-level=info
    container_name: redpanda
    image: docker.redpanda.com/redpandadata/redpanda:latest
    networks:
      default: null
    ports:
      - mode: ingress
        target: 18081
        published: "18081"
        protocol: tcp
      - mode: ingress
        target: 18082
        published: "18082"
        protocol: tcp
      - mode: ingress
        target: 19092
        published: "19092"
        protocol: tcp
      - mode: ingress
        target: 9644
        published: "19644"
        protocol: tcp
    volumes:
      - type: volume
        source: redpanda
        target: /var/lib/redpanda/data
        volume: {}
  redpanda-console:
    command:
      - -c
      - echo "$$CONSOLE_CONFIG_FILE" > /tmp/config.yml; /app/console
    container_name: redpanda-console
    depends_on:
      redpanda:
        condition: service_started
        required: true
    entrypoint:
      - /bin/sh
    environment:
      CONFIG_FILEPATH: /tmp/config.yml
      CONSOLE_CONFIG_FILE: |-
        kafka:
          brokers: ["redpanda:9092"]
          schemaRegistry:
            enabled: true
            urls: ["http://redpanda:8081"]
        redpanda:
          adminApi:
            enabled: true
            urls: ["http://redpanda:9644"]
    image: docker.redpanda.com/redpandadata/console:latest
    networks:
      default: null
    ports:
      - mode: ingress
        target: 8080
        published: "8080"
        protocol: tcp
  zaiko:
    build:
      context: /home/ec2-user/mictlan/apps/zaiko
      dockerfile: Dockerfile
    depends_on:
      mongo:
        condition: service_started
        required: true
      redpanda-console:
        condition: service_started
        required: true
    entrypoint:
      - /entrypoint.sh
    networks:
      default: null
    ports:
      - mode: ingress
        target: 80
        published: "80"
        protocol: tcp
    volumes:
      - type: bind
        source: /home/ec2-user/mictlan/infra/infra/zaiko/entrypoint.sh
        target: /entrypoint.sh
        bind:
          create_host_path: true
      - type: bind
        source: /home/ec2-user/mictlan/infra/infra/zaiko/stockconnector.yaml
        target: /stockconnector.yaml
        bind:
          create_host_path: true
networks:
  default:
    name: mictlan_default
volumes:
  redpanda:
    name: mictlan_redpanda
```
