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

// TagもContentBlockと同様に動的に追加する
type TagId string
type Tag struct {
	Id   TagId
	Name string
}

// StorageKeyがファイル追加後に生成されるためバックエンドロジックでプレースホルダーと実際のファイルとの紐付けを行う必要がある
type Attachment struct {
	Name       string
	Type       string
	Size       int64
	StorageKey string
}
type PlaceHolder string

type QuestionId string
type Question struct {
	Sub          auth.Sub
	Id           QuestionId
	Title        string
	CreatedAt    time.Time
	UpdatedAt    time.Time
	BestAnswerId AnswerId // Solvedの場合はBestAnswerIdに解答IDが入る
	Tags         []Tag

	ContentBlocks []ContentBlock // ContentBlocksにはplaceholderを含んだテキストが入る
	Attachments   map[PlaceHolder]Attachment
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
	attachments map[PlaceHolder]Attachment,
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
		attachments = map[PlaceHolder]Attachment{}
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

type AnswerId string
type Answer struct {
	Sub auth.Sub
	Id  AnswerId
}

func NewAnswer(sub auth.Sub) Answer {
	return Answer{Sub: sub}
}
