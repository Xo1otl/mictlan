# Q&A

このプロジェクトは、Robert C. Martin氏が提唱するClean Architectureの原則に従って構築されています。以下に、各レイヤーとその実装の詳細を示します。

## 1. Domain Layer (Enterprise Business Rules)

Clean Architectureの中核を成す層で、ビジネスロジックとルールを含みます。

### 1.1 Entities

- ファイル: `entities.go`
- 説明: ビジネスオブジェクトを表現する構造体やインターフェースを定義します。
- 例: Question, Answer, Attachment など

### 1.2 Use Cases (Application Business Rules)

- ファイル: `mutation.go`, `query.go`
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