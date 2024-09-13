package qa

import "context"

type MockStorage struct{}

// Put implements Storage.
func (m *MockStorage) Put(ctx context.Context, object *Object) (Attachment, error) {
	panic("unimplemented")
}

func NewMockStorage() Storage {
	return &MockStorage{}
}
