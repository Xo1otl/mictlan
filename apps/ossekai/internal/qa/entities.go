package qa

import (
	"errors"
	"io"
	"maps"
	"ossekaiserver/internal/auth"
	"time"
)

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

var (
	ErrMissingRequiredFields = errors.New("missing required fields")
	ErrEmptyContentBlocks    = errors.New("content blocks cannot be empty")
)

type QuestionId string

type Answer struct {
	Sub auth.Sub
	Id  AnswerId
}

func NewAnswer(sub auth.Sub) Answer {
	return Answer{Sub: sub}
}

type AnswerId string

// stack overflowの場合、タグはDBに登録されている
// 信頼あるユーザーのみタグの作成が可能になっている
// タグの作成は質問をする時に行われる
// それまで存在しないタグを書いても検証はされない
// だからAskQuestionの時にTagNameを受け取ってそのNameのタグが存在するか確かめれば良い
// TagにはNameとIdがあるが、どちらもユニークである
// TagもContentBlockと同様に動的に追加する
type Tag struct {
	Id   TagId
	Name TagName
}

type TagId string
type TagName string

type Attachment struct {
	Placeholder string
	Kind        string
	Size        int64
	ObjectKey   ObjectKey
}

func NewAttachment(placeholder string, kind string, size int64, objectKey ObjectKey) *Attachment {
	// 引数でnilを防いでいる以外に特にvalidationは必要ない...
	return &Attachment{
		Placeholder: placeholder,
		Kind:        kind,
		Size:        size,
		ObjectKey:   objectKey,
	}
}

type ObjectKey string

// Content Aggregateは質問の内容のバリデーションをカプセル化している
type Content struct {
	Objects Objects
	Blocks  ContentBlocks
}

func NewContent(blocks ContentBlocks, objects Objects, parse func(string) ([]string, error)) (*Content, error) {
	op, err := objects.Placeholders()
	if err != nil {
		return nil, err
	}
	bp, err := blocks.Placeholders(parse)
	if err != nil {
		return nil, err
	}
	if !maps.Equal(op, bp) {
		return nil, ErrContentObjectMismatch
	}
	return &Content{
		Blocks:  blocks,
		Objects: objects,
	}, nil
}

var (
	ErrPlaceholderConflict   = errors.New("placeholder conflict")
	ErrContentObjectMismatch = errors.New("contentBlocks and objects mismatch")
)

type ContentBlocks []*ContentBlock

func (c *ContentBlocks) Placeholders(parse func(string) ([]string, error)) (map[string]bool, error) {
	placeholderMap := make(map[string]bool)
	for _, block := range *c {
		placeholders, err := parse(block.Content)
		if err != nil {
			return nil, err
		}
		for _, placeholder := range placeholders {
			placeholderMap[placeholder] = true
		}
	}
	return placeholderMap, nil
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

type Objects []*Object

func (o *Objects) Placeholders() (map[string]bool, error) {
	placeholderMap := make(map[string]bool)
	for _, object := range *o {
		if placeholderMap[object.Placeholder] {
			return nil, ErrPlaceholderConflict
		}
		placeholderMap[object.Placeholder] = true
	}
	return placeholderMap, nil
}

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

type ObjectSrc io.Reader
