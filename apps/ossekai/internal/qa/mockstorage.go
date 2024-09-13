package qa

import (
	"context"
	"io"
	"log"
)

type MockStorage struct{}

// Put implements Storage.
func (m *MockStorage) Put(ctx context.Context, object *Object) (*Attachment, error) {
	data, err := io.ReadAll(object.Src)
	if err != nil {
		return nil, err
	}
	log.Printf("put object: %s", data)
	panic("unimplemented")
}

func NewMockStorage() Storage {
	return &MockStorage{}
}
