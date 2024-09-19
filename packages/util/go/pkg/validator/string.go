package validator

import (
	"errors"
	"path/filepath"
	"regexp"
	"strings"
)

func Tag(parent Validator[string]) Validator[string] {
	v := new(parent)
	v.validate = func() error {
		value, err := v.Value()
		if err != nil {
			return err
		}

		// 長さチェック（1-35文字）とタグの形式チェックを1つの正規表現で行う
		if !regexp.MustCompile(`^(?:[a-z0-9]+[-+])*[a-z0-9]+$`).MatchString(value) {
			return ErrBadTag
		}

		return nil
	}
	parent.Propagate(v)
	return v
}

var ErrBadTag = errors.New("not tag")

func ShortLine(parent Validator[string]) Validator[string] {
	v := new(parent)
	v.validate = func() error {
		value, err := v.Value()
		if err != nil {
			return err
		}
		if len(value) < 5 || len(value) > 100 || strings.Contains(value, "\n") {
			return ErrNotShortLine
		}
		return nil
	}
	parent.Propagate(v)
	return v
}

var ErrNotShortLine = errors.New("not short line")

func VerySimple(parent Validator[string]) Validator[string] {
	v := new(parent)
	v.validate = func() error {
		value, err := v.Value()
		if err != nil {
			return err
		}
		// Check length
		if len(value) < 3 || len(value) > 20 {
			return ErrNotVerySimple
		}
		// Check format using regex
		match, err := regexp.MatchString("^[a-zA-Z0-9-]+$", value)
		if err != nil {
			return err // This should rarely happen, but it's good to handle it
		}
		if !match {
			return ErrNotVerySimple
		}
		return nil
	}
	parent.Propagate(v)
	return v
}

var ErrNotVerySimple = errors.New("not very simple")

// reducer
func Filename(parent Validator[string]) Validator[string] {
	v := new(parent)
	v.reduce = func(s string) (string, error) {
		// filepath.Base関数を使用してパスの最後の要素（ファイル名）を取得
		filename := filepath.Base(s)
		// ファイル名が "." または ".." の場合、エラーを返す
		if filename == "." || filename == ".." {
			return "", ErrInvalidFilename
		}
		// 拡張子を除去
		return strings.TrimSuffix(filename, filepath.Ext(filename)), nil
	}
	parent.Propagate(v)
	return v
}

var ErrInvalidFilename = errors.New("invalid filename")
