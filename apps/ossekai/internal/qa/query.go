package qa

type QueryRepo interface {
}

type Query struct {
	repo QueryRepo
}

func NewQuery(repo QueryRepo) *Query {
	return &Query{repo: repo}
}
