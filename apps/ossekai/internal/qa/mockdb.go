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

// AddQuestion implements Repo.
func (m *MockDb) AddQuestion(sub auth.Sub, title string, tagNames []TagName, contentBlocks []*ContentBlock, attachments []*Attachment) (*QuestionId, error) {
	id := QuestionId(gofakeit.UUID())
	// 実際はtagNamesを使ってTagをDBから取得し、そのIDを含めてtagのEntityを作成する
	// 存在しないtagについてはDBに登録するが、定期的に不適切なtagを削除したり等のバッチ処理が必要
	tags := make([]Tag, len(tagNames))
	for i, tagName := range tagNames {
		tags[i] = Tag{
			Id:   TagId(gofakeit.UUID()),
			Name: tagName,
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
