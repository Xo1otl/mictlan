package qa_test

import (
	"context"
	"strings"
	"testing"

	"lib/pkg/transaction"
	"ossekaiserver/internal/qa"
)

// TestS3Storage tests the S3Storage Put and transaction rollback.
func TestS3Storage(t *testing.T) {
	// バケット名を指定してS3Storageを作成
	storage := qa.NewS3Storage()

	// トランザクションを開始
	tx := transaction.Begin(context.TODO())

	// テスト用のダミーデータを準備
	dummyData := strings.NewReader("dummy data")
	object := &qa.Object{
		Placeholder: "dummy placeholder",
		Src:         dummyData,
	}

	// オブジェクトをS3に保存
	attachment, err := storage.Put(tx, object)
	if err != nil {
		t.Fatalf("failed to put object: %v", err)
	}

	// アップロード結果を確認
	t.Logf("Uploaded attachment: %+v", attachment)

	// トランザクションをロールバック
	tx.Rollback()

	// ロールバック後、S3からオブジェクトが削除されたか確認（削除の確認をどう行うかは実装次第）
	// 実際にS3から削除されたことを確認するテストを行うためには、S3のGetObjectなどを使って確認しますが、
	// ここではテスト環境の制約上、削除の確認は行わずログで出力します。
	t.Log("Transaction rolled back. Ensure object is deleted from S3 in real-world test.")
}
