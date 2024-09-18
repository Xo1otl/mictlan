package qa

import "errors"

// idやkeyはCQRSの文脈でも共通
type QuestionId string
type TagId string
type AnswerId string
type ObjectKey string

// valueObjectはCQRSの文脈でも使い回せる
type Attachment struct {
	Placeholder string
	Kind        string
	Size        int64
	ObjectKey   ObjectKey
}

func NewAttachment(placeholder string, kind string, size int64, objectKey ObjectKey) *Attachment {
	return &Attachment{
		Placeholder: placeholder,
		Kind:        kind,
		Size:        size,
		ObjectKey:   objectKey,
	}
}

// ContentBlockはテキストやマークダウンやlatexをサポートする予定
// 個々のタイプはDomainレイヤで定義する内容ではないし、ハードコードするのではなくデータベースに動的に追加できるようにする
type ContentBlock struct {
	Kind    string
	Content string
}

func NewContentBlock(kind string, content string) (*ContentBlock, error) {
	if kind == "" {
		return nil, ErrEmptyContentKind
	}
	if content == "" {
		return nil, ErrEmptyContentBlock
	}
	return &ContentBlock{Kind: kind, Content: content}, nil
}

var (
	ErrEmptyContentKind  = errors.New("content block type cannot be empty")
	ErrEmptyContentBlock = errors.New("content block content cannot be empty")
)
