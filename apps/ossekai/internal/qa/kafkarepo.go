package qa

import (
	"lib/pkg/transaction"
	"ossekaiserver/internal/auth"
)

type KafkaRepo struct {
}

// AddQuestion implements CommandRepo.
func (k *KafkaRepo) AddQuestion(tx transaction.Transaction, sub auth.Sub, title Title, tagIds []TagId, contentBlocks []*ContentBlock, attachments []*Attachment) (*QuestionId, error) {
	panic("unimplemented")
}

// DefineTags implements CommandRepo.
func (k *KafkaRepo) DefineTags(tx transaction.Transaction, tags []CustomTag) ([]TagId, error) {
	return nil, nil
}

func NewKafkaRepo() CommandRepo {
	return &KafkaRepo{}
}
