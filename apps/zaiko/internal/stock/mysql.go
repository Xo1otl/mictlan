package stock

import "github.com/shopspring/decimal"

type MySQL struct {
}

// Sales implements Repo.
func (m *MySQL) Sales() decimal.Decimal {
	panic("unimplemented")
}

// Stocks implements Repo.
func (m *MySQL) Stocks(name string) map[string]int {
	panic("unimplemented")
}

func NewMySQL() Repo {
	return &MySQL{}
}
