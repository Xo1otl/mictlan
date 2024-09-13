package qa

import (
	"context"
	"ossekaiserver/internal/auth"
)

type App struct {
	repo    Repo
	storage Storage
}

func NewApp(repo Repo, storage Storage) *App {
	return &App{repo, storage}
}

func (a *App) AskQuestion(sub auth.Sub, title string, tagIds []TagId, contentBlocks []*ContentBlock, objects []*Object) (*QuestionId, error) {
	// TODO: ContentBlockの中身を解析して不正なplaceholderがないかチェックする
	attachments := make(Attachments, len(objects))
	for i, object := range objects {
		attachment, err := a.storage.Put(context.TODO(), object)
		if err != nil {
			return nil, err
		}
		attachments[i] = attachment
	}
	questionId, err := a.repo.AddQuestion(sub, title, tagIds, contentBlocks, attachments)
	if err != nil {
		return nil, err
	}
	return questionId, nil
}

func (a *App) Answers(questionId QuestionId) []Answer {
	return a.repo.Answers(questionId)
}
