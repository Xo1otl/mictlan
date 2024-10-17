package stock

import "github.com/shopspring/decimal"

type Repo interface {
	Stocks(name string) map[string]int
	Sales() decimal.Decimal
}
