package validator_test

import (
	"testing"
	"util/pkg/validator"
)

func TestString(t *testing.T) {
	v := validator.Filename(validator.New("helloworld.txt"))
	validator.VerySimple(v)
	validator.ShortLine(v)
	if err := v.Validate(); err != nil {
		t.Fatal(err)
	}
	if _, err := v.Value(); err != nil {
		t.Fatal(err)
	}
}
