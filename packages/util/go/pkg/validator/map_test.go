package validator_test

import (
	"testing"
	"util/pkg/validator"
)

func TestMap(t *testing.T) {
	v := validator.FilenameFromKeys(validator.New(map[string]struct{}{"helloworld.txt.txt": {}, "goodbye.txt.txt": {}}))
	v = validator.FilenameFromKeys(v)
	err := validator.AllKeysAreVerySimple(v).Validate()
	if err != nil {
		t.Fatal(err)
	}
}
