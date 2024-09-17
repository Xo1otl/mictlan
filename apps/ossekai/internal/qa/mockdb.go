package qa

import (
	"encoding/json"
	"errors"
	"fmt"
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

// DefineTags implements CommandRepo.
func (m *MockDb) DefineTags(customTags []CustomTag) ([]TagId, error) {
	id := TagId(gofakeit.UUID())
	tags := make([]Tag, 0, len(customTags))
	tagIds := make([]TagId, 0, len(customTags))
	for _, ct := range customTags {
		tags = append(tags, NewTag(id, ct.Name))
		tagIds = append(tagIds, id)
	}
	m.tables.Tags = append(m.tables.Tags, tags...)
	m.save()
	return tagIds, nil
}

// AddQuestion implements Repo.
func (m *MockDb) AddQuestion(sub auth.Sub, title string, tagIds []TagId, contentBlocks []*ContentBlock, attachments []*Attachment) (*QuestionId, error) {
	id := QuestionId(gofakeit.UUID())
	tags := make([]Tag, 0, len(tagIds))
	for _, tagId := range tagIds {
		for _, tag := range m.tables.Tags {
			if tag.Id == tagId {
				tags = append(tags, tag)
				break
			}
		}
	}
	question, err := NewQuestion(sub, id, title, time.Now(), time.Now(), "", tags, contentBlocks, attachments)
	if err != nil {
		return nil, err
	}
	m.tables.Questions = append(m.tables.Questions, question)
	m.save()
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
