package validator_test

import (
	"testing"
	"util/pkg/validator"
)

func TestValidator(t *testing.T) {
	v := validator.Filename(validator.New("aaaaaaaaaaaaaaa.ttt"))
	validator.ShortText(validator.ShortText(validator.VerySimple(validator.ShortText(v))))
	err := v.Validate()
	t.Log(err)
}
