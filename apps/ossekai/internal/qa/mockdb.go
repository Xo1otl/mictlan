package qa

import (
	"encoding/json"
	"errors"
	"fmt"
	"lib/pkg/transaction"
	"log"
	"os"
	"ossekaiserver/internal/auth"
	"path/filepath"
	"sync"
	"time"

	"github.com/brianvoe/gofakeit/v7"
)

type Tables struct {
	Questions []*Question
	Tags      []Tag
}

type MockDb struct {
	mu       sync.Mutex
	tables   Tables
	filePath string
}

func NewMockDb() CommandRepo {
	filePath := filepath.Join(os.TempDir(), "mockdb.json")
	db := &MockDb{
		tables:   Tables{},
		filePath: filePath,
	}
	db.load()
	return db
}

func NewMockDbAsQueryRepo() QueryRepo {
	filePath := filepath.Join(os.TempDir(), "mockdb.json")
	db := &MockDb{
		tables:   Tables{},
		filePath: filePath,
	}
	db.load()
	return db
}

// FindQuestionByTitle implements QueryRepo.
func (m *MockDb) FindQuestionByTitle(title string) ([]*Question, error) {
	panic("unimplemented")
}

// FindTagByName implements QueryRepo.
func (m *MockDb) FindTagByName(name string) (*Tag, error) {
	m.load()
	for _, tag := range m.tables.Tags {
		if tag.Name == name {
			return &tag, nil
		}
	}
	return nil, errors.New("tag not found")
}

// DefineTags implements CommandRepo.
func (m *MockDb) DefineTags(tx transaction.Transaction, customTags []CustomTag) ([]TagId, error) {
	tags := make([]Tag, 0, len(customTags))
	tagIds := make([]TagId, 0, len(customTags))
	for _, ct := range customTags {
		for _, tag := range m.tables.Tags {
			if tag.Name == ct.Name {
				return nil, errors.New("tag already exists")
			}
		}
		id := TagId(gofakeit.UUID())
		tags = append(tags, NewTag(id, ct.Name))
		tagIds = append(tagIds, id)
		log.Print("Defined tag: ", ct.Name)
	}
	// 現在のタグの状態を保存しておく
	originalTags := make([]Tag, len(m.tables.Tags))
	copy(originalTags, m.tables.Tags)

	// 新しいタグを追加
	m.tables.Tags = append(m.tables.Tags, tags...)

	transaction.WithRollback(tx, func() {
		// 何か問題があればロールバックし、元の状態に戻す
		m.tables.Tags = originalTags
		log.Println("Transaction rolled back, restored original tags")
	})
	transaction.WithCommit(tx, func() { m.save() })
	return tagIds, nil
}

// AddQuestion implements Repo.
func (m *MockDb) AddQuestion(tx transaction.Transaction, sub auth.Sub, title Title, tagIds []TagId, contentBlocks []*ContentBlock, attachments []*Attachment) (*QuestionId, error) {
	id := QuestionId(gofakeit.UUID())
	tags := make([]Tag, 0, len(tagIds))
	for _, tagId := range tagIds {
		for _, tag := range m.tables.Tags {
			if tag.Id == tagId {
				tags = append(tags, tag)
				log.Print("Added predefined tag: ", tag.Name)
				break
			}
		}
	}
	if len(tags) != len(tagIds) {
		return nil, errors.New("tag not found")
	}
	question, err := NewQuestion(sub, id, string(title), time.Now(), time.Now(), "", tags, contentBlocks, attachments)
	if err != nil {
		return nil, err
	}
	// 現在の質問の状態を保存しておく
	originalQuestions := make([]*Question, len(m.tables.Questions))
	copy(originalQuestions, m.tables.Questions)

	// 新しい質問を追加
	m.tables.Questions = append(m.tables.Questions, question)

	transaction.WithRollback(tx, func() {
		// 何か問題があればロールバックし、元の状態に戻す
		m.tables.Questions = originalQuestions
		log.Println("Transaction rolled back, restored original questions")
	})

	transaction.WithCommit(tx, func() { m.save() })
	return &id, nil
}

// Answers implements Repo.
func (m *MockDb) Answers(questionId QuestionId) []Answer {
	count := 10
	answers := make([]Answer, count)
	gofakeit.Slice(&answers)
	return answers
}

// Question implements Repo.
func (m *MockDb) Question(questionId QuestionId) (*Question, error) {
	for _, question := range m.tables.Questions {
		if question.Id == questionId {
			return question, nil
		}
	}
	return nil, errors.New("question not found")
}

func (m *MockDb) load() {
	m.mu.Lock()
	defer m.mu.Unlock()

	file, err := os.Open(m.filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return
		}
		fmt.Println("Error opening file:", err)
		return
	}
	defer file.Close()

	var tables Tables
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&tables)
	if err != nil {
		fmt.Println("Error decoding JSON:", err)
	}
	m.tables = tables
}

func (m *MockDb) save() {
	m.mu.Lock()
	defer m.mu.Unlock()

	file, err := os.Create(m.filePath)
	if err != nil {
		fmt.Println("Error creating file:", err)
		return
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	err = encoder.Encode(m.tables)
	if err != nil {
		fmt.Println("Error encoding JSON:", err)
	}
}
