package qa

type MockStorage struct{}

// Put implements Storage.
func (m *MockStorage) Put(date []byte) (string, error) {
	panic("unimplemented")
}

func NewMockStorage() Storage {
	return &MockStorage{}
}
