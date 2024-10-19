package stock_test

import (
	"testing"
	"zaiko/internal/stock"
)

func TestMongoRepo(t *testing.T) {
	repo := stock.NewMongoRepo()
	sales := repo.Sales("testusersub3")
	stocks := repo.Stocks("testusersub3", "")
	t.Log(sales)
	t.Log(stocks)
}
