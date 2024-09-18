package qa

import "lib/pkg/transaction"

type S3Storage struct {
}

// Put implements CommandStorage.
func (s *S3Storage) Put(tx transaction.Transaction, object *Object) (*Attachment, error) {
	panic("unimplemented")
}

func NewS3Storage() CommandStorage {
	return &S3Storage{}
}
