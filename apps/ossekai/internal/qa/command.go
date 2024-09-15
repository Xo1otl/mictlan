package qa

import (
	"context"
	"errors"
	"ossekaiserver/internal/auth"
)

type Command struct {
	repo    CommandRepo
	storage CommandStorage
}

func NewCommand(repo CommandRepo, storage CommandStorage) *Command {
	return &Command{repo, storage}
}

func (a *Command) AskQuestion(sub auth.Sub, title string, tagNames []TagName, content *Content) (*QuestionId, error) {
	attachments := make([]*Attachment, len(content.Objects))
	for i, object := range content.Objects {
		attachment, err := a.storage.Put(context.TODO(), object)
		if err != nil {
			return nil, err
		}
		attachments[i] = attachment
	}
	questionId, err := a.repo.AddQuestion(sub, title, tagNames, content.Blocks, attachments)
	if err != nil {
		return nil, err
	}
	return questionId, nil
}

type CommandRepo interface {
	AddQuestion(sub auth.Sub, title string, tagNames []TagName, contentBlocks []*ContentBlock, attachments []*Attachment) (*QuestionId, error)
}

type CommandStorage interface {
	// ObjectKeyの生成はだいたいUUID等が必要だけどこれをapplication layerで行うにはgoogle/uuidなどの抽象化が必要
	// これはめんどくさすぎるけど、ストレージの実装はinfraを使用できるレイヤのためそこで行えば抽象化は不用
	// そのため、Storageはuuidを引数で受け取らない。これはawsのs3のPutObjectとは仕様が異なる
	// 同様にしてobjectからattachmentへの変換も過程でファイルタイプの判定などが存在するが、このインターフェースではその実装を暗に要求する
	Put(ctx context.Context, object *Object) (*Attachment, error)
}

var (
	ErrEmptyObjectPlaceholder = errors.New("object placeholder cannot be empty")
	ErrNilObjectSrc           = errors.New("object src cannot be nil")
)
