package stock

import (
	"fmt"
	"log"
	"net/http"
	"zaiko/internal/auth"

	"github.com/labstack/echo/v4"
	"github.com/shopspring/decimal"
)

func AddEchoRoutes(e *echo.Echo) {
	eventStore := &InMemoryEventStore{}
	command := NewCommand(eventStore, eventStore)
	repo := NewInMemoryRepo(eventStore)
	ch := NewCommandHandler(command)
	qh := NewQueryHandler(repo)

	route := e.Group("/v1")
	route.Use(setMockClaims)

	route.POST("/stocks", ch.HandleStocks)
	route.DELETE("/stocks", ch.HandleStocks)
	route.POST("/sales", ch.HandleSales)

	route.GET("/stocks", qh.HandleStocks)
	route.GET("/sales", qh.HandleSales)
}

func setMockClaims(next echo.HandlerFunc) echo.HandlerFunc {
	return func(c echo.Context) error {
		claims := &auth.Claims{
			Sub: "testusersub",
		}
		c.Set("claims", claims)
		return next(c)
	}
}

type CommandHandler struct {
	command *Command
}

func NewCommandHandler(command *Command) *CommandHandler {
	return &CommandHandler{command}
}

func (h *CommandHandler) HandleStocks(c echo.Context) error {
	claims := c.Get("claims")
	if claims == nil {
		return c.JSON(http.StatusUnauthorized, map[string]string{"error": "claims not found"})
	}
	sub := claims.(*auth.Claims).Sub
	switch c.Request().Method {
	case http.MethodPost:
		req := struct {
			Name   string `json:"name"`
			Amount int    `json:"amount"`
		}{}
		if err := c.Bind(&req); err != nil {
			log.Println("failed to parse request body:", err)
			return c.JSON(http.StatusBadRequest, map[string]string{"message": "ERROR"})
		}
		if req.Name == "" || req.Amount <= 0 {
			log.Println("invalid name or amount")
			return c.JSON(http.StatusBadRequest, map[string]string{"message": "ERROR"})
		}
		if err := h.command.Add(sub, req.Name, req.Amount); err != nil {
			log.Println("command add error:", err)
			return c.JSON(http.StatusBadRequest, map[string]string{"message": "ERROR"})
		}
		return c.JSON(http.StatusOK, req)
	case http.MethodDelete:
		h.command.ClearAll(sub)
		return c.NoContent(http.StatusOK)
	default:
		return c.JSON(http.StatusMethodNotAllowed, map[string]string{"message": "Method Not Allowed"})
	}
}

func (h *CommandHandler) HandleSales(c echo.Context) error {
	claims := c.Get("claims")
	if claims == nil {
		return c.JSON(http.StatusUnauthorized, map[string]string{"error": "claims not found"})
	}
	sub := claims.(*auth.Claims).Sub
	if c.Request().Method != http.MethodPost {
		return c.JSON(http.StatusMethodNotAllowed, map[string]string{"message": "Method Not Allowed"})
	}
	req := struct {
		Name   string  `json:"name"`
		Amount *int    `json:"amount,omitempty"`
		Price  float64 `json:"price,omitempty"`
	}{}
	if err := c.Bind(&req); err != nil {
		log.Println("failed to parse request body:", err)
		return c.JSON(http.StatusBadRequest, map[string]string{"message": "ERROR"})
	}
	if req.Name == "" || req.Price < 0 {
		log.Println("invalid name or price")
		return c.JSON(http.StatusBadRequest, map[string]string{"message": "ERROR"})
	}
	if req.Price <= 0 {
		log.Println("Price is zero")
	}
	// Amountは省略可能で、省略された場合は1として扱うが、明示的に0を指定するのは不正
	amount := 1
	if req.Amount != nil {
		if *req.Amount == 0 {
			// 0かどうかのチェックはnilでない時のみ行う
			log.Println("Amount is zero, which is invalid")
			return c.JSON(http.StatusBadRequest, map[string]string{"message": "ERROR"})
		}
		amount = *req.Amount
	}
	if err := h.command.Sell(sub, req.Name, amount, decimal.NewFromFloat(req.Price)); err != nil {
		log.Println("command sell error:", err)
		return c.JSON(http.StatusBadRequest, map[string]string{"message": "ERROR"})
	}
	return c.JSON(http.StatusOK, req)
}

type QueryHandler struct {
	repo Repo
}

func NewQueryHandler(repo Repo) *QueryHandler {
	return &QueryHandler{repo}
}

func (h *QueryHandler) HandleStocks(c echo.Context) error {
	if c.Request().Method != http.MethodGet {
		return c.JSON(http.StatusMethodNotAllowed, map[string]string{"message": "Method Not Allowed"})
	}
	stocks := h.repo.Stocks("")
	return c.JSON(http.StatusOK, stocks)
}

func (h *QueryHandler) HandleSales(c echo.Context) error {
	if c.Request().Method != http.MethodGet {
		return c.JSON(http.StatusMethodNotAllowed, map[string]string{"message": "Method Not Allowed"})
	}

	sales, exact := h.repo.Sales().Float64()
	if !exact {
		log.Println("セールスは厳密な値ではありません")
	}
	response := SalesResponse{
		Sales: Float64(sales),
	}

	return c.JSON(http.StatusOK, response)
}

type SalesResponse struct {
	Sales Float64 `json:"sales"`
}

type Float64 float64

func (f Float64) MarshalJSON() ([]byte, error) {
	// 例にあった480.0や400.0は小数点第一位までしかないが、480.000000002のような値は480.00にする必要があるのか確認
	return []byte(fmt.Sprintf("%.1f", f)), nil
}
