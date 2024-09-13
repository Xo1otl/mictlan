package qa_test

import (
	"testing"
	"time"

	"ossekaiserver/internal/auth"
	"ossekaiserver/internal/qa"
)

func TestNewQuestion(t *testing.T) {
	validSub := auth.Sub("user123")
	validId := qa.QuestionId("q1")
	validTitle := "How to write Go tests?"
	validTime := time.Now()
	validContentBlocks := []qa.ContentBlock{{"text", "This is a test question"}}

	tests := []struct {
		name          string
		sub           auth.Sub
		id            qa.QuestionId
		title         string
		createdAt     time.Time
		updatedAt     time.Time
		bestAnswerId  qa.AnswerId
		tags          []qa.Tag
		contentBlocks []qa.ContentBlock
		attachments   []qa.Attachment
		wantErr       error
	}{
		{
			name:          "Valid question",
			sub:           validSub,
			id:            validId,
			title:         validTitle,
			createdAt:     validTime,
			updatedAt:     validTime,
			bestAnswerId:  "",
			tags:          nil,
			contentBlocks: validContentBlocks,
			attachments:   nil,
			wantErr:       nil,
		},
		{
			name:          "Missing sub",
			sub:           "",
			id:            validId,
			title:         validTitle,
			createdAt:     validTime,
			updatedAt:     validTime,
			contentBlocks: validContentBlocks,
			wantErr:       qa.ErrMissingRequiredFields,
		},
		{
			name:          "Missing id",
			sub:           validSub,
			id:            "",
			title:         validTitle,
			createdAt:     validTime,
			updatedAt:     validTime,
			contentBlocks: validContentBlocks,
			wantErr:       qa.ErrMissingRequiredFields,
		},
		{
			name:          "Missing title",
			sub:           validSub,
			id:            validId,
			title:         "",
			createdAt:     validTime,
			updatedAt:     validTime,
			contentBlocks: validContentBlocks,
			wantErr:       qa.ErrMissingRequiredFields,
		},
		{
			name:          "Zero createdAt",
			sub:           validSub,
			id:            validId,
			title:         validTitle,
			createdAt:     time.Time{},
			updatedAt:     validTime,
			contentBlocks: validContentBlocks,
			wantErr:       qa.ErrMissingRequiredFields,
		},
		{
			name:          "Empty content blocks",
			sub:           validSub,
			id:            validId,
			title:         validTitle,
			createdAt:     validTime,
			updatedAt:     validTime,
			contentBlocks: []qa.ContentBlock{},
			wantErr:       qa.ErrEmptyContentBlocks,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := qa.NewQuestion(tt.sub, tt.id, tt.title, tt.createdAt, tt.updatedAt, tt.bestAnswerId, tt.tags, tt.contentBlocks, tt.attachments)
			if err != tt.wantErr {
				t.Errorf("NewQuestion() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
