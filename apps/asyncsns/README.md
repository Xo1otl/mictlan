# asyncsns
- awsの一次面接で提出するプロジェクト

## プレゼン

(かっこ内の文章は読まない)

### (ドメインの説明)
- 本日は、テキストや写真・動画によりユーザーの非同期コミュニケーションを実現するSNSの構成について発表させていただきます
- このSNSプラットフォームは、メディアを含むコンテンツの投稿、リアルタイムチャット、高度な検索、プッシュ通知、通話、ライブストリーミングなど、主要なSNS機能をすべて備えた包括的なシステムとなっています

### (アーキテクチャーの説明)
- まず、アーキテクチャの全体像についてご説明します
- お手元の構成図では、システム内の処理の流れを、副作用を持たないリードオンリーなクエリを破線、コマンドを実線で表しています
- 両方が存在する場合は太線で表され、リアルタイムコミュニケーションのためのコネクションは双方向矢印で表現しています
- また、機能ごとにフレームで区切って示しております
- それでは各機能について、詳しくご説明させていただきます

### (CDN)
- CDNの役割は、コンテンツの効率的な配信とアクセス制御とグローバル展開です
- 例えば、ssl証明書の一元管理や、Origin Access Controlポリシーに基づいたS3へのアクセス制御を、CloudFrontやWafで実装します

### (Gateway)
- Gatewayの役割は、APIリクエストの統合的な制御と管理です
- 例えば、jwtの一元的な検証、lambdaの統合、ステートフルエンドポイント、apiバージョンの切り替え、負荷対策などの機能をAPI Gatewayで実装します

### (UI)
- UIの役割は、フロントエンドです
- S3を用いてホストします

### (Auth)
- Authの役割は、ユーザー認証とアクセス制御の一元管理です
- Cognito User Poolでユーザー管理とJWTベースの認証を実装し、Identity Poolを通じて一時的なAWSクレデンシャルを発行することで、セキュアなリソースアクセスを実現します

### (API)
- APIの役割は、システムの中核となるドメインロジックです
- コマンド処理apiでは、投稿、編集、リアルタイムチャット、通話などの処理を、Text Repository、Media Repository、RTC、Notificationなどのサービスと連携して実現します
    - 例えば投稿処理では、Text RepositoryとMedia Repositoryとnotificationを使って、メディアファイルをアップロードし、テキストデータを保存して、友達に通知を送るといった一連の処理を行います
    - また、リアルタイムチャットでは、Repositoryとrtcとnotificationを使って、接続情報を管理しながらメッセージをリアルタイムやり取りし、オフラインのユーザーには通知を送るといった処理を行います
- クエリapiでは、ユーザーが必要とする情報をRepositoryから取得します
    - 例えば、投稿に対する全文検索や、友達のプロフィール画像の取得などが可能です

### (Text Repository)
- Text Repositoryの役割は、テキスト情報の永続化です
- dynamodbで保存したデータをopensearchでzero ETL統合することで、高度な検索機能を実現します

### (Media Repository)
- Media Repositoryの役割は、ファイルの永続化です
- S3とデフォルトのSSE-S3暗号化とS3 Triggerを使用した署名付きURLによる共有を利用することで、大容量メディアファイルをセキュアに保存・配信します

### (RTC)
- RTCの役割は、リアルタイムコミュニケーションの実現です
- websocket apiによるリアルタイムチャットと、IVSによる通話やライブストリーミング機能を提供します
- レプリケーションを行うことで、Active Standbyやマルチリージョン対応を行います

### (Notification)
- Notificationの役割は、プッシュ通知の配信です
- Amazon SNSで実装し、デッドレターキューで失敗時の分析や再処理に対応します

### (Observability)
- Observabilityの役割は、システム全体の監視と分析です
- CloudWatchによるシステム性能メトリクスの収集・監視、X-Rayによる分散トレーシング、CloudTrailによるAPIやリソースの操作履歴の記録・監査、ManagedGrafanaによる統合的な可視化とダッシュボード作成を行います
- これにより、障害時の原因特定やボトルネックの特定、システムの状態管理を行います

### (Secrets)
- Secretsの役割は、機密情報の一元管理です
- Parameter Storeを用いて設定値や構成データを管理し、SecretsManagerを用いてデーターベースの認証情報やAPIキーなどの機密情報をローテーション管理します

### (まとめ)
- 以上が、本SNSサービスの構成となります
- マネージドサービスをフル活用することで、インフラ運用の手間を減らし、従量課金でコストを最適化し、高可用性とトラフィックに応じた自動スケーリングとグローバル展開を実現します
- セキュリティ面では、Waf、Cognito、SecretsManager、Parameter Storeなどによる一元的な保護を行い、異常発生時には豊富な観測手段による問題特定が可能です

## Q&A
- CloudFrontの機能について
    - エッジキャッシング
    - セキュリティ
        - DDoS保護、WAF、OACによるS3の保護、圧縮、HTTP/3対応
        - アクセスログ解析
- データベースへの書き込みが遅くなるケースの例
    - 投稿がバズった時
- 管理者、開発者、エンドユーザーがそれぞれどのリソースにアクセスするのか
    - 開発者
        - AWS CDKによるインフラストラクチャのデプロイ権限
        - CloudWatchログの閲覧・分析権限
        - テスト環境の全リソースへのフルアクセス
        - 本番環境への制限付きアクセス
    - 管理者
        - AWS Consoleへのアクセス権限
        - ユーザー管理（Cognito）
        - セキュリティ設定の管理
    - エンドユーザー
        - CDN経由でAPIやUIにアクセス
        - 署名付きURLによるS3コンテンツアクセス
        - WebSocket APIへの接続
        - Cognitoによる認証・認可
- api gatewayが停止したらどうなるか
    - フロントエンドは見れる
    - 認証は動く
    - 署名されたファイルのリンクを一時的に持ってるユーザーだけアクセスできる
    - それ以外のすべての機能は動かない
- dynamoDB選定理由
    - auroraとかと迷ったけど、日付の範囲と全文検索とかユースケース考えたらdynamoとopensearchがいいかなと思った
    - 投稿テーブル: パーティションキー=UserID, ソートキー=Timestamp
    - GSIというのがあるらしい
    - デノーマライズ
- なぜs3
    - 業界標準だから
    - 安い
    - マネージド
- lambdaのコールドスタート問題
    - provisioned concurrency機能であっためとく
- 通知システムでのsqs/sns選択理由
    
### 1. DynamoDBへの書き込みが遅くなるケースとその対策

Q: どのような場合にDynamoDBへの書き込みが遅くなり、それをどう対処しますか？

A: 主に以下のケースが考えられます：

1. ホットパーティション問題
   - 特定のパーティションキーに書き込みが集中する場合
   - 対策：
     - パーティションキーの設計見直し（ユーザーIDに時間要素を付加等）
     - Write Sharding（ランダムサフィックス付加）の実装
     - Adaptive Capacity機能の活用

2. プロビジョニング容量の不足
   - 対策：
     - Auto Scalingの適切な設定
     - On-Demand容量モードの検討
     - CloudWatchメトリクスによる早期検知

3. 大量の同時書き込み
   - 対策：
     - SQSによる書き込みのバッファリング
     - BatchWriteItemの活用
     - 書き込みの分散化

### 2. DynamoDBを選択した理由

Q: なぜRDBMSではなくDynamoDBを選択したのですか？

A: 以下の理由から選択しました：

1. スケーラビリティ
   - SNSの特性上、データ量が予測不能
   - サーバーレスアーキテクチャとの親和性
   - Auto Scaling機能による柔軟な対応

2. パフォーマンス
   - 一貫した低レイテンシー（シングルミリ秒）
   - グローバルテーブルによる地理的分散
   - インデックス設計の柔軟性

3. コスト最適化
   - 使用量に応じた課金
   - プロビジョニングの柔軟な調整
   - 運用コストの削減

### その他の想定質問と回答

1. データモデリング関連

Q: DynamoDBでのSNSデータモデリングの具体的な設計について説明してください。

A:
- 投稿テーブル：パーティションキー=UserID、ソートキー=Timestamp
- タイムラインテーブル：パーティションキー=UserID、ソートキー=Timestamp
- GSIを活用したアクセスパターン最適化
- デノーマライズによる読み取り効率化

2. 一貫性関連

Q: Eventually Consistentな特性がSNSにもたらす影響とその対策は？

A:
- タイムライン更新の遅延許容
- Strong Consistencyが必要な操作の識別
- 重要な操作での一貫性レベルの選択
- バージョニングによる矛盾解決

3. バックアップと災害対策

Q: データの永続性と可用性をどのように確保していますか？

A:
- Point-in-time Recovery (PITR)の活用
- グローバルテーブルによる地理的冗長性
- バックアップの自動化とテスト
- 障害シナリオの定期的な検証

Q: dynamodbのレイテンシが高いときどうする

A: 
- 平均レイテンシを確認する
- DAXを有効化する

4. 監視とパフォーマンスチューニング

Q: システムのパフォーマンスをどのように監視・最適化していますか？

A:
- CloudWatchメトリクスの活用
- X-Rayによるトレーシング
- キャパシティプランニングの定期的な見直し
- ホットパーティションの検知と対応

5. コスト最適化

Q: DynamoDBのコストを最適化するための戦略は？

A:
- 適切なキャパシティモードの選択
- TTLによる不要データの自動削除
- リザーブドキャパシティの活用
- アクセスパターンに基づくインデックス設計

## Note
- APIがCQRSによって二つに分類される
- 図はコマンドに焦点あてて書いてる
- Queryの場合、Lambda不要になる部分がある
    - opensearchやs3は直接データ読みに行く
    - lambdaのconsumerでsqsをconsumeして通知を配信する(queryかどうかも怪しい)
- 図を書く (まとめの役割)
- api gatewayやx-ray等の、複数にまたがるサービスから順に、広いほうから説明
- ドメインから考えていく、各ドメインは自身のコンテキストに集中できるよう、カプセル化された説明
- 抽象的な説明 -> 具体や詳細 -> まとめで抽象に戻る (qiskitの説明動画のやり方参考にする)

## refs
- [chat application](https://aws.amazon.com/jp/blogs/news/building-a-full-stack-chat-application-with-aws-and-nextjs/)
- [api gateway](https://docs.aws.amazon.com/ja_jp/apigateway/latest/developerguide/http-api-jwt-authorizer.html)
- [localstack](https://docs.docker.com/guides/localstack/)
- [aws cdk local](https://github.com/localstack/aws-cdk-local?tab=readme-ov-file)
- [oac](https://qiita.com/shota_hagiwara/items/caacbda7f55aeea110d1)
- [ライブ配信](https://docs.aws.amazon.com/ja_jp/ivs/latest/RealTimeUserGuide/obs-whip-support.html)
- [DB比較](https://dynobase.dev/dynamodb-vs-aurora/)
- [CloudFront](https://aws.amazon.com/blogs/aws/new-http-3-support-for-amazon-cloudfront/)
- [provisioned concurrency](https://dev.classmethod.jp/articles/lambda-provisioned-concurrency-coldstart/)
- [provisioned concurrency cost](https://dev.classmethod.jp/articles/simulate-provisioned-concurrency-cost/)
- [amazon sns](https://docs.aws.amazon.com/sns/latest/dg/welcome.html)
- [dynamodb opensearch](https://qiita.com/Yodeee/items/415b11e0e886ec93ec8a)
- [dynamodb opensearch official](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/OpenSearchIngestionForDynamoDB.html)
- [opensearch](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/configure-client-ddb.html)
- [opensearch serverless](https://qiita.com/neruneruo/items/d4dd391147e0af709d4e)
- [zero etl](https://aws.amazon.com/jp/what-is/zero-etl/)
- [amazon sns dlq](https://docs.aws.amazon.com/sns/latest/dg/sns-dead-letter-queues.html)
- [webrtc amazon](https://docs.aws.amazon.com/kinesisvideostreams-webrtc-dg/latest/devguide/what-is-kvswebrtc.html)
- [amazon kinesis video streams](https://docs.aws.amazon.com/kinesisvideostreams/latest/dg/what-is-kinesis-video.html)
- [amazon sns resend dlq](https://docs.aws.amazon.com/ja_jp/sns/latest/dg/sns-message-delivery-retries.html)
- [route53 secondary](https://docs.aws.amazon.com/ja_jp/apigateway/latest/developerguide/disaster-recovery-resiliency.html)
- [api gateway api種類](https://docs.aws.amazon.com/ja_jp/apigateway/latest/developerguide/api-gateway-api-endpoint-types.html)
- [x ray](https://docs.aws.amazon.com/xray/latest/devguide/xray-sdk-go-configuration.html)
- [aws waf](https://docs.aws.amazon.com/waf/latest/developerguide/getting-started.html)
- [dynamodb high latency](https://repost.aws/ja/knowledge-center/dynamodb-high-latency)
- [dynamodb用のcache](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/DAX.html)
- [athena grafana s3](https://aws.amazon.com/blogs/big-data/visualize-amazon-s3-data-using-amazon-athena-and-amazon-managed-grafana/)
- [github ivs example](https://aws.amazon.com/blogs/media/add-multiple-hosts-to-live-streams-with-amazon-ivs/)
- [IVS Configure Thumbnail](https://docs.aws.amazon.com/ivs/latest/LowLatencyAPIReference/API_ThumbnailConfiguration.html)
