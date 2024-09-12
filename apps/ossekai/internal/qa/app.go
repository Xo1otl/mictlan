package qa

import "context"

type App struct {
	Repo
	Storage
}

func NewApp(repo Repo, storage Storage) *App {
	return &App{repo, storage}
}

// FIXME: 受け取るのはQuestionではない
// AttachmentDataとObjectKey以外のサーバーで検証/生成しないデータを受け取る
func (a *App) AskQuestion(q QuestionInput, objects Objects) {
	a.Storage.PutObjects(context.TODO(), objects)
	// TODO: Attachmentの処理
	// 1. ObjectKeyを使ってStorageにPut
	// 2. ObjectKeyをAttachmentにセット
	// 3. データをrepoで保存
	// a.Repo.AddQuestion(q)
}

func (a *App) Answers(q Question) []Answer {
	return a.Repo.Answers(q)
}
