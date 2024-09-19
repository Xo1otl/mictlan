// map関連のvalidationは全部ここにまとめる
package validator

// validator
func AllKeysAreVerySimple(parent Validator[map[string]struct{}]) Validator[map[string]struct{}] {
	v := new(parent)
	v.validate = func() error {
		m, err := v.Value()
		if err != nil {
			return err
		}
		for k := range m {
			if err := VerySimple(New(k)).Validate(); err != nil {
				return err
			}
		}
		return nil
	}
	parent.Propagate(v)
	return v
}

// reducer
func FilenameFromKeys(parent Validator[map[string]struct{}]) Validator[map[string]struct{}] {
	v := new(parent)
	v.reduce = func(m map[string]struct{}) (map[string]struct{}, error) {
		m2 := make(map[string]struct{})
		for k := range m {
			filename, err := Filename(New(k)).Value()
			if err != nil {
				return nil, err
			}
			m2[filename] = struct{}{}
		}
		return m2, nil
	}
	parent.Propagate(v)
	return v
}
