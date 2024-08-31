package qa

type App struct {
	Repo
}

func NewApp(repo Repo) *App {
	return &App{repo}
}

func (a *App) AskQuestion(q Question) {
	a.Repo.AskQuestion(q)
}

func (a *App) Answers(q Question) []Answer {
	return a.Repo.Answers(q)
}
