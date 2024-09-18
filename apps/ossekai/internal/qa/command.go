package qa

import (
	"context"
	"errors"
	"io"
	"lib/pkg/transaction"
	"maps"
	"ossekaiserver/internal/auth"
	"strings"
	"util/pkg/validator"
)

type Command struct {
	repo    CommandRepo
	storage CommandStorage
}

func NewCommand(repo CommandRepo, storage CommandStorage) *Command {
	return &Command{repo, storage}
}

// TODO: subだけでなくclaimsからtagの追加が許可されたユーザーかどうかの判定が必要
func (a *Command) AskQuestion(sub auth.Sub, title Title, tagSet *TagSet, content *Content) (*QuestionId, error) {
	tx := transaction.Begin(context.TODO())
	defer tx.Rollback()
	attachments := make([]*Attachment, 0, len(content.Objects))
	for _, object := range content.Objects {
		attachment, err := a.storage.Put(tx, object)
		if err != nil {
			return nil, err
		}
		attachments = append(attachments, attachment)
	}
	// TODO: stachoverflowみたいにタグの定義の許可の仕組みを作る
	// primary keyの重複等はrepositoryの責務
	definedTagIds, err := a.repo.DefineTags(tx, tagSet.Custom)
	if err != nil {
		return nil, err
	}
	tagIds := tagSet.Predefined
	tagIds = append(tagIds, definedTagIds...)
	questionId, err := a.repo.AddQuestion(tx, sub, title, tagIds, content.Blocks, attachments)
	if err != nil {
		return nil, err
	}
	if err = tx.Commit(); err != nil {
		return nil, err
	}
	return questionId, nil
}

type CommandRepo interface {
	AddQuestion(tx transaction.Transaction, sub auth.Sub, title Title, tagIds []TagId, contentBlocks []*ContentBlock, attachments []*Attachment) (*QuestionId, error)
	DefineTags(tx transaction.Transaction, tags []CustomTag) ([]TagId, error)
}

type CommandStorage interface {
	// ObjectKeyの生成はだいたいUUID等が必要だけどこれをapplication layerで行うにはgoogle/uuidなどの抽象化が必要
	// これはめんどくさすぎるけど、ストレージの実装はinfraを使用できるレイヤのためそこで行えば抽象化は不用
	// そのため、Storageはuuidを引数で受け取らない。これはawsのs3のPutObjectとは仕様が異なる
	// 同様にしてobjectからattachmentへの変換も過程でファイルタイプの判定などが存在するが、このインターフェースではその実装を暗に要求する
	Put(tx transaction.Transaction, object *Object) (*Attachment, error)
}

type Title string

func NewTitle(title string) (*Title, error) {
	v := validator.ShortText(validator.New(title))
	if err := v.Validate(); err != nil {
		return nil, err
	}
	t := Title(title)
	return &t, nil
}

type TagSet struct {
	Predefined []TagId
	Custom     []CustomTag
}

func NewTagSet(predefinedIds []string, customNames []string) (*TagSet, error) {
	var custom []CustomTag
	for _, name := range customNames {
		if ct, err := NewCustomTag(name); err != nil {
			return nil, err
		} else {
			custom = append(custom, *ct)
		}
	}
	var predefined []TagId
	for _, id := range predefinedIds {
		predefined = append(predefined, TagId(id))
	}
	seenPredefined := make(map[TagId]struct{})
	seenCustom := make(map[CustomTag]struct{})
	for _, tag := range predefined {
		if _, exists := seenPredefined[tag]; exists {
			return nil, ErrTagIdConflict
		}
		seenPredefined[tag] = struct{}{}
	}
	for _, tag := range custom {
		if _, exists := seenCustom[tag]; exists {
			return nil, ErrCustomTagNameConflict
		}
		seenCustom[tag] = struct{}{}
	}
	tagSet := TagSet{Predefined: predefined, Custom: custom}
	return &tagSet, nil
}

var (
	ErrTagIdConflict         = errors.New("tag conflict")
	ErrCustomTagNameConflict = errors.New("custom tag conflict")
)

type CustomTag struct {
	Name string
}

func NewCustomTag(name string) (*CustomTag, error) {
	name = strings.TrimSpace(name)
	v := validator.VerySimple(validator.New(name))
	if err := v.Validate(); err != nil {
		return nil, err
	}
	return &CustomTag{Name: name}, nil
}

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

func (c *ContentBlocks) Placeholders(parse func(string) ([]string, error)) (map[string]struct{}, error) {
	placeholderMap := make(map[string]struct{})
	for _, block := range *c {
		placeholders, err := parse(block.Content)
		if err != nil {
			return nil, err
		}
		for _, placeholder := range placeholders {
			placeholderMap[placeholder] = struct{}{}
		}
	}
	return placeholderMap, nil
}

type Objects []*Object

func (o *Objects) Placeholders() (map[string]struct{}, error) {
	placeholderMap := make(map[string]struct{})
	for _, object := range *o {
		_, exists := placeholderMap[object.Placeholder]
		if exists {
			return nil, ErrPlaceholderConflict
		}
		placeholderMap[object.Placeholder] = struct{}{}
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

var (
	ErrEmptyObjectPlaceholder = errors.New("object placeholder cannot be empty")
	ErrNilObjectSrc           = errors.New("object src cannot be nil")
)
