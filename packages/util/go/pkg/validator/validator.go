package validator

type Validatable any

type Validator[T Validatable] interface {
	Validate() error
	Value() (T, error)
	Propagate(Validator[T])
}

func New[T Validatable](o T) Validator[T] {
	return &validator[T]{
		validate: func() error { return nil },
		reduce:   func(T) (T, error) { return o, nil },
	}
}

type validator[T Validatable] struct {
	parent   Validator[T]
	children []Validator[T]
	validate func() error
	reduce   func(T) (T, error)
}

func new[T Validatable](v Validator[T]) *validator[T] {
	return &validator[T]{parent: v}
}

func (v *validator[T]) Value() (T, error) {
	var pv T
	var err error
	if v.parent != nil {
		if pv, err = v.parent.Value(); err != nil {
			return pv, err
		}
	}
	if v.reduce != nil {
		return v.reduce(pv)
	}
	return pv, nil
}

func (v *validator[T]) Validate() error {
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

func (v *validator[T]) Propagate(child Validator[T]) {
	v.children = append(v.children, child)
}
