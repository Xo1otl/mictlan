package qa_test

import (
	"ossekaiserver/internal/qa"
	"testing"
	"util/pkg/validator"
)

func TestNewCustomTag(t *testing.T) {
	tests := []struct {
		name    string
		tagName string
		wantErr error
	}{
		{"Valid tag", "valid-tag123", nil},
		{"Empty tag", "", validator.ErrNotVerySimple},
		{"Too short tag", "ab", validator.ErrNotVerySimple},
		{"Too long tag", "this-tag-name-is-way-too-long-and-should-not-be-allowed", validator.ErrNotVerySimple},
		{"Invalid characters", "invalid!@#", validator.ErrNotVerySimple},
		{"Valid edge case (3 chars)", "a-1", nil},
		{"Valid edge case (20 chars)", "12345678901234567890", nil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := qa.NewCustomTag(tt.tagName)
			if err != tt.wantErr {
				t.Errorf("NewCustomTag() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if err == nil && got.Name != tt.tagName {
				t.Errorf("NewCustomTag() got = %v, want %v", got.Name, tt.tagName)
			}
		})
	}
}
