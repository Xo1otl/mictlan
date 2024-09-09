package qa

type App struct {
	Repo
	Storage
}

func NewApp(repo Repo, storage Storage) *App {
	return &App{repo, storage}
}

func (a *App) AskQuestion(q Question) {
	a.Repo.AskQuestion(q)
}

func (a *App) Answers(q Question) []Answer {
	return a.Repo.Answers(q)
}
