package qa

import (
	"context"
	"errors"
	"ossekaiserver/internal/auth"
)

type MutationRepo interface {
	AddQuestion(sub auth.Sub, title string, tagNames []TagName, contentBlocks []*ContentBlock, attachments []*Attachment) (*QuestionId, error)
}

var (
	ErrEmptyObjectPlaceholder = errors.New("object placeholder cannot be empty")
	ErrNilObjectSrc           = errors.New("object src cannot be nil")
)

type MutationStorage interface {
	// ObjectKeyの生成はだいたいUUID等が必要だけどこれをapplication layerで行うにはgoogle/uuidなどの抽象化が必要
	// これはめんどくさすぎるけど、ストレージの実装はinfraを使用できるレイヤのためそこで行えば抽象化は不用
	// そのため、Storageはuuidを引数で受け取らない。これはawsのs3のPutObjectとは仕様が異なる
	// 同様にしてobjectからattachmentへの変換も過程でファイルタイプの判定などが存在するが、このインターフェースではその実装を暗に要求する
	Put(ctx context.Context, object *Object) (*Attachment, error)
}

type Mutation struct {
	repo    MutationRepo
	storage MutationStorage
}

func NewMutation(repo MutationRepo, storage MutationStorage) *Mutation {
	return &Mutation{repo, storage}
}

func (a *Mutation) AskQuestion(sub auth.Sub, title string, tagNames []TagName, contentBlocks []*ContentBlock, objects []*Object) (*QuestionId, error) {
	// TODO: ContentBlockの中身を解析して不正なplaceholderがないかチェックする
	attachments := make([]*Attachment, len(objects))
	for i, object := range objects {
		attachment, err := a.storage.Put(context.TODO(), object)
		if err != nil {
			return nil, err
		}
		attachments[i] = attachment
	}
	questionId, err := a.repo.AddQuestion(sub, title, tagNames, contentBlocks, attachments)
	if err != nil {
		return nil, err
	}
	return questionId, nil
}
