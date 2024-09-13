package qa

import (
	"context"
	"errors"
	"io"
	"ossekaiserver/internal/auth"
)

type Repo interface {
	// id, createdAt, updatedAtはadapterで生成する
	AddQuestion(sub auth.Sub, title string, tagIds []TagId, contentBlocks []*ContentBlock, attachments Attachments) (*QuestionId, error)
	// TODO: Question単体で取得するより、検索等によって複数取得する場合が多い気がするので考えなおす
	Question(questionId QuestionId) Question
	Answers(questionId QuestionId) []Answer
}

var (
	ErrEmptyObjectPlaceholder = errors.New("object placeholder cannot be empty")
)

type ObjectData io.Reader
type Object struct {
	Placeholder string
	Data        ObjectData
}

func NewObject(placeholder string, data ObjectData) (*Object, error) {
	if placeholder == "" {
		return nil, ErrEmptyObjectPlaceholder
	}
	return &Object{
		Placeholder: placeholder,
		Data:        data,
	}, nil
}

type ObjectKey string

type Storage interface {
	// ObjectKeyの生成はだいたいUUID等が必要だけどこれをapplication layerで行うにはgoogle/uuidなどの抽象化が必要
	// これはめんどくさすぎるけど、ストレージの実装はinfraを使用できるレイヤのためそこで行えば抽象化は不用
	// そのため、Storageはuuidを引数で受け取らない。これはawsのs3のPutObjectとは仕様が異なる
	// 同様にしてobjectからattachmentへの変換も過程でファイルタイプの判定などが存在するが、このインターフェースではその実装を暗に要求する
	Put(ctx context.Context, object *Object) (Attachment, error)
}
