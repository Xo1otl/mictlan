# FunSearch Cloud Run Deployment

このディレクトリには、FunSearchアプリケーションをGoogle Cloud Runにデプロイするための設定が含まれています。

## 前提条件

1. **Google Cloud SDK** インストール済み
2. **Terraform** インストール済み  
3. **Docker** インストール済み
4. **Google Cloud認証**設定済み

## デプロイ手順

### 1. イメージのビルド・プッシュ
```bash
# Google Cloud認証
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Artifact Registry認証
gcloud auth configure-docker asia-northeast1-docker.pkg.dev

# イメージビルド・プッシュ
docker build -t asia-northeast1-docker.pkg.dev/YOUR_PROJECT_ID/funsearch/funsearch:latest .
docker push asia-northeast1-docker.pkg.dev/YOUR_PROJECT_ID/funsearch/funsearch:latest
```

### 2. インフラストラクチャのデプロイ
```bash
cd deployments/terraform/

# 環境変数設定
export TF_VAR_google_cloud_api_key="your-gemini-api-key"

# Terraform実行
terraform init
terraform plan -var="project_id=YOUR_PROJECT_ID"
terraform apply -var="project_id=YOUR_PROJECT_ID"
```

### 3. IAP認証の設定（手動）
```bash
# Cloud RunでIAP有効化
gcloud beta run deploy funsearch \
  --region=asia-northeast1 \
  --image=asia-northeast1-docker.pkg.dev/YOUR_PROJECT_ID/funsearch/funsearch:latest \
  --iap

# IAPアクセス権限付与（プロジェクトオーナーが実行）
gcloud iap web add-iam-policy-binding \
  --resource-type=cloud-run \
  --service=funsearch \
  --region=asia-northeast1 \
  --member="domain:qunasys.com" \
  --role="roles/iap.httpsResourceAccessor"
```

## 設定概要

### Terraform管理リソース
- Artifact Registry リポジトリ
- Cloud Run サービス
- 必要なAPI有効化（Cloud Run API, Artifact Registry API）

### 手動設定項目
- **IAP有効化**: `gcloud beta run deploy --iap`で設定
- **ユーザーアクセス権限**: `run.services.setIamPolicy`権限を持つ管理者が設定

### 認証方式
- **IAP認証**: Google認証によるアクセス制御
- **Gradio Basic認証**: IAP使用時は無効化済み

## 更新手順
```bash
# 1. 新しいイメージをビルド・プッシュ
docker build -t asia-northeast1-docker.pkg.dev/YOUR_PROJECT_ID/funsearch/funsearch:latest .
docker push asia-northeast1-docker.pkg.dev/YOUR_PROJECT_ID/funsearch/funsearch:latest

# 2. Cloud Runサービス更新
terraform apply -var="project_id=YOUR_PROJECT_ID"
```

## Terraform設定ファイル

- `main.tf`: メインのインフラ設定
- `variables.tf`: 変数定義  
- `outputs.tf`: 出力値定義

## IAP設定について

### 現在の方式（混合）
- **Terraform**: 基本インフラ管理
- **手動**: IAP有効化とアクセス権限設定

### Terraformでの完全自動化について
Google CloudドキュメントによるとTerraformでもIAP設定可能ですが、Load Balancer経由の複雑な構成が必要です。
- 参考: https://cloud.google.com/iap/docs/enabling-cloud-run#terraform
- 現在はシンプルさを優先し手動設定を採用

## トラブルシューティング

### 権限エラーの場合
`run.services.setIamPolicy`権限が必要です。プロジェクトオーナーまたは`Cloud Run Admin`ロールを持つユーザーに依頼してください。

### URLの確認
```bash
terraform output service_url
```

# TODO:
* GitHubの連携とかやっていいらしいので、コンテナのビルド自動化など行う、cloud build使う
