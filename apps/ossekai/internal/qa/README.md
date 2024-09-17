# Q&A

このプロジェクトは、Robert C. Martin氏が提唱するClean Architectureの原則に従って構築されています。以下に、各レイヤーとその実装の詳細を示します。

## CQRS

- AskQuestion等のCommandと読み取りを分離している
- 分離ができるているかの検証方法
	- 外側のレイヤのコードをすべてコメントアウト
	- エラーがないことを確認
	- query.goかcommand.goをすべてコメントアウト
	- どちらの場合でもエラーがないことを確認

## 1. Domain Layer (Enterprise Business Rules)

Clean Architectureの中核を成す層で、ビジネスロジックとルールを含みます。

### 1.1 Entities

- ファイル: `entities.go`
- 説明: ビジネスオブジェクトを表現する構造体やインターフェースを定義します。
- 例: Question, Answer, Attachment など

### 1.2 Use Cases (Application Business Rules)

- ファイル: `command.go`, `query.go`
- 説明: アプリケーション固有のビジネスルールを実装します。エンティティを操作し、ビジネスロジックを実行します。
- 例: AskQuestion, SearchQuestions など

## 2. Interface Adapters

外部インターフェースとドメインレイヤーの間の変換を担当します。

### 2.1 Controllers/Presenters

- ファイル: `echoroute.go`
- 説明: HTTPリクエストを受け取り、Use Casesを呼び出し、レスポンスを整形します。

### 2.2 Gateways

- ファイル: `mockdb.go`, `mockstorage.go`
- 説明: データベースやストレージとのインターフェースを提供します。実際の実装はInfrastructure Layerで行われます。

## 3. Frameworks & Drivers (Infrastructure)

外部フレームワークやツールとの統合を担当します。

### 3.1 Web Framework

- 使用技術: Echo
- 説明: HTTPリクエストのルーティングとハンドリングを行います。

### 3.2 Database

- 使用技術: DynamoDB (モック実装)
- 説明: データの永続化を担当します。

### 3.3 External Services

- 使用技術: S3 (モック実装)
- 説明: ファイルストレージなどの外部サービスとの連携を行います。

## 依存関係の方向

Clean Architectureの重要な原則である「依存関係の方向」を遵守しています：

1. Domain Layer (Entities, Use Cases) は外部のレイヤーに依存しません。
2. Interface Adapters は Domain Layer に依存しますが、Frameworks & Drivers には依存しません。
3. Frameworks & Drivers は内側のレイヤーに依存します。

この構造により、ビジネスロジックが外部の実装の詳細から隔離され、テスト容易性、保守性、拡張性が向上します。

## テスト戦略

各レイヤーは独立してテスト可能です：

1. Domain Layer: 単体テストでビジネスロジックを検証
2. Interface Adapters: モックを使用して Use Cases との統合をテスト
3. Frameworks & Drivers: 実際の外部サービスを使用した統合テスト

このClean Architecture実装により、コードの責務が明確に分離され、長期的なプロジェクトの保守性と拡張性が確保されます。

## データの形式

```json
{
	"Questions": [
		{
			"Sub": "238bf3a2-0bc2-41db-abdb-e880558589f1",
			"Id": "d4dcf735-6783-487b-ae29-0ffaf1c84206",
			"Title": "Your these failure with she.",
			"CreatedAt": "2024-09-15T07:55:30.324534036Z",
			"UpdatedAt": "2024-09-15T07:55:30.324534097Z",
			"BestAnswerId": "",
			"Tags": [
				{ "Id": "7613b687-48c1-48bf-9309-802e946603e8", "Name": "SiMPLE" }
			],
			"ContentBlocks": [
				{
					"Kind": "latex",
					"Content": "Brother consequently which her lately app speed huh product time. At throughout mine empty animal quarterly apart you it either. Weep gee itself frequently everything all one whose this recently. ![Tripdoes] ![Coveycould]"
				},
				{
					"Kind": "latex",
					"Content": "Book generosity why clumsy of they huh roll another appetite. These tonight shall these been when Burkinese horse since across. Of does yesterday anyone hers school twist besides life next. ![LawnGreengirl] ![Libraryhas] ![Violetair]"
				},
				{
					"Kind": "markdown",
					"Content": "It yourself publicity she below work hmm hourly them staff. Block monthly Beninese us these win wash her anger annually. Truth despite even that tonight of Turkmen on it muster."
				}
			],
			"Attachments": [
				{
					"Placeholder": "Tripdoes",
					"Kind": "application/octet-stream",
					"Size": 27,
					"ObjectKey": "858845e7-aca0-48f5-9a56-a4f9a8e8e494"
				},
				{
					"Placeholder": "Coveycould",
					"Kind": "application/octet-stream",
					"Size": 30,
					"ObjectKey": "6739d24b-5049-46da-9e62-26ba551a8a67"
				},
				{
					"Placeholder": "LawnGreengirl",
					"Kind": "application/octet-stream",
					"Size": 37,
					"ObjectKey": "4443d8a2-f78f-4df2-affb-b7bcea205251"
				},
				{
					"Placeholder": "Libraryhas",
					"Kind": "application/octet-stream",
					"Size": 30,
					"ObjectKey": "a13f2216-71af-42d9-8aba-48c066dc4469"
				},
				{
					"Placeholder": "Violetair",
					"Kind": "application/octet-stream",
					"Size": 32,
					"ObjectKey": "5a7069af-ece5-49b5-aef9-3ecdc9a6d184"
				}
			]
		},
		{
			"Sub": "4c5e69d9-0e98-43ae-b06f-1774e372ea82",
			"Id": "1b55a316-9f51-4881-8d24-1a4bf04d62d3",
			"Title": "Child where still yearly previously.",
			"CreatedAt": "2024-09-15T07:55:50.558655318Z",
			"UpdatedAt": "2024-09-15T07:55:50.558655368Z",
			"BestAnswerId": "",
			"Tags": [
				{ "Id": "22e7cf8b-0f4a-4d9c-a39d-c4b469a25466", "Name": "A++" },
				{ "Id": "55ca9fa0-7646-45b2-a062-f3d23a3929c6", "Name": "Reia" }
			],
			"ContentBlocks": [
				{
					"Kind": "markdown",
					"Content": "These single those information alternatively sedge somebody spin behalf his. This fact since where its bravo many why dynasty of. Above whose should Hindu deskpath firstly Intelligent our what brilliance."
				},
				{
					"Kind": "text",
					"Content": "Yesterday crew long horse weekly nap i.e. wealth few whenever. Previously yikes alternatively who that what give close those bow. Daily anyway enough that until in yoga gently deer weekly."
				},
				{
					"Kind": "latex",
					"Content": "Moreover which still where wake that place many speed nobody. Straightaway freezer brass then soon their instead since to whose. Fragile stand say Polish firstly case crawl these his finally. ![Orangefrailty]"
				},
				{
					"Kind": "markdown",
					"Content": "Below occasionally heavy neither packet those could often sometimes suit. Island that that its whoever for were these towards elsewhere. Jittery even completely stand without yet backwards us this hand. ![Goldfishshall]"
				},
				{
					"Kind": "latex",
					"Content": "Normally nevertheless other consequently now time say swallow path that. Why Gaussian remain secondly him constantly childhood aggravate to those. Till my crew virtually yours earlier what backwards envy generally."
				}
			],
			"Attachments": [
				{
					"Placeholder": "Orangefrailty",
					"Kind": "application/octet-stream",
					"Size": 22,
					"ObjectKey": "4556182d-a688-46cc-8633-6154bbd08574"
				},
				{
					"Placeholder": "Goldfishshall",
					"Kind": "application/octet-stream",
					"Size": 27,
					"ObjectKey": "8944cb3d-dbc7-4344-b1ce-2baa47363fe7"
				}
			]
		}
	]
}
```