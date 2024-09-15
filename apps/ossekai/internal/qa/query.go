package qa

type Query struct {
	repo QueryRepo
}

func NewQuery(repo QueryRepo) *Query {
	return &Query{repo: repo}
}

func (q *Query) FindQuestionByTitle(title string) ([]*Question, error) {
	return q.repo.FindByTitle(title)
}

func (q *Query) SearchQuestion(id QuestionId) ([]*Question, error) {
	// TODO: llmでsqlのクエリ生成して検索, ベクトルデータベースを使用して検索, GraphRAG,
	return nil, nil
}

type QueryRepo interface {
	FindByTitle(title string) ([]*Question, error)
}
