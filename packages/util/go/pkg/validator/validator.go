package validator

import (
	"errors"
	"regexp"
	"strings"
)

type Validator interface {
	Validate() error
	Value() string
	Propagate(Validator)
}

type validator struct {
	parent   Validator
	children []Validator
	validate func() error
	reduce   func(s string) string
}

func New(s string) Validator {
	return &validator{
		validate: func() error { return nil },
		reduce:   func(_ string) string { return s },
	}
}

func (v *validator) Value() string {
	pv := ""
	if v.parent != nil {
		pv = v.parent.Value()
	}
	if v.reduce != nil {
		return v.reduce(pv)
	}
	return pv
}

func (v *validator) Validate() error {
	for _, child := range v.children {
		if err := child.Validate(); err != nil {
			return err
		}
	}
	if v.validate == nil {
		return nil
	}
	if err := v.validate(); err != nil {
		return err
	}
	return nil
}

func (v *validator) Propagate(child Validator) {
	v.children = append(v.children, child)
}

func VerySimple(parent Validator) Validator {
	child := &validator{parent: parent}
	child.validate = func() error {
		if !verySimpleRegex.MatchString(parent.Value()) {
			return ErrNotVerySimple
		}
		return nil
	}
	parent.Propagate(child)
	return child
}

var (
	ErrNotVerySimple = errors.New("not very simple")
	verySimpleRegex  = regexp.MustCompile(`^[a-zA-Z0-9-]{3,20}$`)
)

func ShortText(parent Validator) Validator {
	child := &validator{parent: parent}
	child.validate = func() error {
		s := parent.Value()
		if len(s) < 5 || len(s) > 100 {
			return ErrNotShortText
		}
		return nil
	}
	parent.Propagate(child)
	return child
}

var (
	ErrNotShortText = errors.New("not short title")
)

func Filename(parent Validator) Validator {
	child := &validator{parent: parent}
	child.reduce = func(s string) string {
		return strings.Split(s, ".")[0]
	}
	parent.Propagate(child)
	return child
}
