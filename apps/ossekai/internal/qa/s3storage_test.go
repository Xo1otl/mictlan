package qa_test

import (
	"context"
	"lib/pkg/transaction"
	"ossekaiserver/internal/qa"
	"strings"
	"testing"
)

func TestS3Storage(t *testing.T) {
	storage := qa.NewS3Storage()
	tx := transaction.Begin(context.TODO())

	dummyData := strings.NewReader("dummy data")
	object, err := qa.NewObject("dummy placeholder", dummyData)
	if err != nil {
		t.Fatal(err)
	}

	attachment, err := storage.Put(tx, object)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(attachment)
}
