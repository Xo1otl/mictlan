package qa

import (
	"fmt"
	"io"
	"lib/pkg/transaction"
	"log"
	"os"
	"path/filepath"

	"github.com/brianvoe/gofakeit/v7"
)

type MockStorage struct {
	basePath string
}

func NewMockStorage() CommandStorage {
	return &MockStorage{
		basePath: filepath.Join(os.TempDir(), "mock_storage"),
	}
}

// Put implements Storage.
func (m *MockStorage) Put(tx transaction.Transaction, object *Object) (*Attachment, error) {
	// 元のファイルパスを保存
	var originalFilePath string

	// ロールバック処理を定義
	transaction.WithRollback(tx, func() {
		if originalFilePath != "" {
			os.Remove(originalFilePath)
			log.Println("Transaction rolled back, removed file:", originalFilePath)
		}
	})

	// Ensure the base directory exists
	if err := os.MkdirAll(m.basePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create base directory: %w", err)
	}

	// Generate a UUID for the filename
	filename := gofakeit.UUID()
	filePath := filepath.Join(m.basePath, filename)
	originalFilePath = filePath

	// Create the file
	file, err := os.Create(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	// Copy the content from the source to the file
	size, err := io.Copy(file, object.Src)
	if err != nil {
		return nil, fmt.Errorf("failed to write file content: %w", err)
	}

	// Determine the MIME type (in a real implementation, you'd use a proper MIME type detection)
	mimeType := "application/octet-stream"

	attachment := NewAttachment(object.Placeholder, mimeType, size, ObjectKey(filename))

	return attachment, nil
}
