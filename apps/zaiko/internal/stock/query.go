package stock

import (
	"zaiko/internal/auth"

	"github.com/shopspring/decimal"
)

type Repo interface {
	Stocks(sub auth.Sub, name string) map[string]int
	Sales(sub auth.Sub) decimal.Decimal
}
