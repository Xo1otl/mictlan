package qa

import (
	"errors"
	"io"
	"ossekaiserver/internal/auth"
	"time"
)

// ContentBlockはテキストやマークダウンやlatexをサポートする予定
// 個々のタイプはDomainレイヤで定義する内容ではないし、ハードコードするのではなくデータベースに動的に追加できるようにする
type ContentBlock struct {
	Kind    string
	Content string
}

var (
	ErrEmptyContentKind  = errors.New("content block type cannot be empty")
	ErrEmptyContentBlock = errors.New("content block content cannot be empty")
)

func NewContentBlock(kind string, content string) (*ContentBlock, error) {
	if kind == "" {
		return nil, ErrEmptyContentKind
	}
	if content == "" {
		return nil, ErrEmptyContentBlock
	}
	return &ContentBlock{Kind: kind, Content: content}, nil
}

type QuestionId string
type Question struct {
	Sub           auth.Sub
	Id            QuestionId
	Title         string
	CreatedAt     time.Time
	UpdatedAt     time.Time
	BestAnswerId  AnswerId // Solvedの場合はBestAnswerIdに解答IDが入る
	Tags          []Tag
	ContentBlocks []*ContentBlock // ContentBlocksにはplaceholderを含んだテキストが入る
	Attachments   []*Attachment
}

var (
	ErrMissingRequiredFields = errors.New("missing required fields")
	ErrEmptyContentBlocks    = errors.New("content blocks cannot be empty")
)

func NewQuestion(
	sub auth.Sub,
	id QuestionId,
	title string,
	createdAt time.Time,
	updatedAt time.Time,
	bestAnswerId AnswerId,
	tags []Tag,
	contentBlocks []*ContentBlock,
	attachments []*Attachment,
) (*Question, error) {
	// Check for nil required fields
	if sub == "" || id == "" || title == "" || createdAt.IsZero() || updatedAt.IsZero() {
		return nil, ErrMissingRequiredFields
	}

	// Check if contentBlocks is empty
	if len(contentBlocks) == 0 {
		return nil, ErrEmptyContentBlocks
	}

	// Initialize empty slices/maps if nil
	if tags == nil {
		tags = make([]Tag, 0)
	}
	if attachments == nil {
		attachments = make([]*Attachment, 0)
	}

	return &Question{
		Sub:           sub,
		Id:            id,
		Title:         title,
		CreatedAt:     createdAt,
		UpdatedAt:     updatedAt,
		BestAnswerId:  bestAnswerId,
		Tags:          tags,
		ContentBlocks: contentBlocks,
		Attachments:   attachments,
	}, nil
}

// stack overflowの場合、タグはDBに登録されている
// 信頼あるユーザーのみタグの作成が可能になっている
// タグの作成は質問をする時に行われる
// それまで存在しないタグを書いても検証はされない
// だからAskQuestionの時にTagNameを受け取ってそのNameのタグが存在するか確かめれば良い
// TagにはNameとIdがあるが、どちらもユニークである

// TagもContentBlockと同様に動的に追加する
type TagId string
type TagName string
type Tag struct {
	Id   TagId
	Name TagName
}

type AnswerId string
type Answer struct {
	Sub auth.Sub
	Id  AnswerId
}

func NewAnswer(sub auth.Sub) Answer {
	return Answer{Sub: sub}
}

type ObjectSrc io.Reader
type Object struct {
	Placeholder string
	Src         ObjectSrc
}

func NewObject(placeholder string, src ObjectSrc) (*Object, error) {
	if placeholder == "" {
		return nil, ErrEmptyObjectPlaceholder
	}
	if src == nil {
		return nil, ErrNilObjectSrc
	}
	return &Object{
		Placeholder: placeholder,
		Src:         src,
	}, nil
}

type ObjectKey string

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
