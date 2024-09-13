package qa

import (
	"errors"
	"ossekaiserver/internal/auth"
	"time"
)

// ContentBlockはテキストやマークダウンやlatexをサポートする予定
// 個々のタイプはDomainレイヤで定義する内容ではないし、ハードコードするのではなくデータベースに動的に追加できるようにする
type ContentBlock struct {
	Type    string
	Content string
}

var (
	ErrEmptyContentType  = errors.New("content block type cannot be empty")
	ErrEmptyContentBlock = errors.New("content block content cannot be empty")
)

func NewContentBlock(t string, c string) (*ContentBlock, error) {
	if t == "" {
		return nil, ErrEmptyContentType
	}
	if c == "" {
		return nil, ErrEmptyContentBlock
	}
	return &ContentBlock{Type: t, Content: c}, nil
}

// TagもContentBlockと同様に動的に追加する
type TagId string
type Tag struct {
	Id   TagId
	Name string
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
	ContentBlocks []ContentBlock // ContentBlocksにはplaceholderを含んだテキストが入る
	Attachments   Attachments
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
	contentBlocks []ContentBlock,
	attachments []Attachment,
) (Question, error) {
	// Check for nil required fields
	if sub == "" || id == "" || title == "" || createdAt.IsZero() || updatedAt.IsZero() {
		return Question{}, ErrMissingRequiredFields
	}

	// Check if contentBlocks is empty
	if len(contentBlocks) == 0 {
		return Question{}, ErrEmptyContentBlocks
	}

	// Initialize empty slices/maps if nil
	if tags == nil {
		tags = []Tag{}
	}
	if attachments == nil {
		attachments = []Attachment{}
	}

	return Question{
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
