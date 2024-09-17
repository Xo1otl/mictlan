package qa

import (
	"errors"
	"ossekaiserver/internal/auth"
	"time"
)

type Query struct {
	// Read側のアプリケーションではRepositoryを呼び出すだけの処理が多いため埋め込む
	QueryRepo
}

func NewQuery(repo QueryRepo) *Query {
	return &Query{QueryRepo: repo}
}

func (q *Query) SearchQuestion(id QuestionId) ([]*Question, error) {
	// TODO: llmでsqlのクエリ生成して検索, ベクトルデータベースを使用して検索, GraphRAG,
	return nil, nil
}

type QueryRepo interface {
	FindTagByName(name string) (*Tag, error)
	FindQuestionByTitle(title string) ([]*Question, error)
}

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

type Answer struct {
	Sub auth.Sub
	Id  AnswerId
}

func NewAnswer(sub auth.Sub) Answer {
	return Answer{Sub: sub}
}

type Tag struct {
	Id   TagId
	Name string
}

func NewTag(id TagId, name string) Tag {
	return Tag{Id: id, Name: name}
}
