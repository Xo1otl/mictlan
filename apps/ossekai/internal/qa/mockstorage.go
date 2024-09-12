package qa

type MockStorage struct{}

// Put implements Storage.
func (m *MockStorage) Put(date Object) (ObjectKey, error) {
	panic("unimplemented")
}

func NewMockStorage() Storage {
	return &MockStorage{}
}
