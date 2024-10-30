# asyncsns

- awsの一次面接で提出するプロジェクト
- 非同期コミュニケーションが可能なsnsをawsで実装する

## coding style

- 図を書く (まとめの役割)
- api gatewayやx-ray等の、複数にまたがるサービスから順に、広いほうから説明
- ドメインから考えていく、各ドメインは自身のコンテキストに集中できるよう、カプセル化された説明
- 抽象的な説明 -> 具体や詳細 -> まとめで抽象に戻る (qiskitの説明動画のやり方参考にする)

# modules

## SSL証明書管理 & cdn

- ssl証明書管理 & cdn
- nginxとかcloudflareみたいな感じ
- cloudfrontを使う

## 認可を行うGateway

- API Gatewayを使用
- lambdaを呼ぶとき等に使う
- cognitoと連携してjwtの検証を行う
- gateway配下のバックエンドでは、jwt検証ロジックを実装しなくていい
- ついでにAPIのバージョン切り替え・負荷対策等

## ログインなどにより「誰であるか」の署名を発行する機能

- Cognitoを使用
- 認証機能 (認可とちょっと違う)
- claim発行する

## テキストデータを永続化するRepository

- DynamoDB
- OpenSearch

## 通知やデータのリアルタイム配信

- チャットではWebSocket API
- それ以外の、TLの通知などはSNS + SQS

## ファイルを永続化するStorage

- S3を使用
- ファイルを保存

## 一部のメディアファイルに事前処理かませたい

- discord等で行われていること
- 動画に対する処理
- MediaConverterを使用

## idea

- chat applicationのを参考にする、
- 動画に対してmedia convert等を挟んで画質の調整などする
- フロントエンドはs3でホストしてcloudfrontで証明書管理等行う
- kmsを使って暗号化する

## diagram

```mermaid
```

## refs

- [chat application](https://aws.amazon.com/jp/blogs/news/building-a-full-stack-chat-application-with-aws-and-nextjs/)
- [localstack](https://docs.docker.com/guides/localstack/)
- [aws cdk local](https://github.com/localstack/aws-cdk-local?tab=readme-ov-file)
