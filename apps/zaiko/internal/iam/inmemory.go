package iam

import (
	"errors"
	"sync"
)

type InMemoryAccountRepo struct {
}

// FindAccount implements AccountRepo.
func (i *InMemoryAccountRepo) FindAccount(username string) (*Account, error) {
	if username == "admin" {
		return NewAccount("1", "admin", "password"), nil
	}
	return nil, errors.New("account not found")
}

func NewInMemoryAccountRepo() AccountRepo {
	return &InMemoryAccountRepo{}
}

// InMemoryDigestRepo struct
type InMemoryDigestRepo struct {
	ncStore map[string]string // nc (nonce count) を保持する
	mu      sync.RWMutex      // Read/Write lock to handle concurrent access
}

// NewInMemoryDigestNcRepo returns a new instance of InMemoryDigestRepo
func NewInMemoryDigestNcRepo() DigestNcRepo {
	return &InMemoryDigestRepo{
		ncStore: make(map[string]string),
	}
}

// get retrieves the get value for a given username
func (i *InMemoryDigestRepo) get(username string) string {
	i.mu.RLock()
	defer i.mu.RUnlock()

	// Return the nc value for the given username, or an empty string if not found
	if nc, ok := i.ncStore[username]; ok {
		return nc
	}
	return ""
}

// set sets the nc value for a given username
func (i *InMemoryDigestRepo) set(username, nc string) error {
	i.mu.Lock()
	defer i.mu.Unlock()

	// Ensure username and nc are not empty
	if username == "" || nc == "" {
		return errors.New("username or nc cannot be empty")
	}

	// Set the nc value for the username
	i.ncStore[username] = nc
	return nil
}
