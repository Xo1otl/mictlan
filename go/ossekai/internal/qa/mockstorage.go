package qa

type MockStorage struct{}

// Delete implements Storage.
func (m *MockStorage) Delete() error {
	panic("unimplemented")
}

// Retrieve implements Storage.
func (m *MockStorage) Retrieve() ([]byte, error) {
	panic("unimplemented")
}

// Store implements Storage.
func (m *MockStorage) Store(data []byte) error {
	panic("unimplemented")
}

func NewMockStorage() Storage {
	return &MockStorage{}
}
