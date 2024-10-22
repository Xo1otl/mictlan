package main

import (
	"log"
	"zaiko/internal/stock"
)

func main() {
	err := stock.CreateTopics()
	if err != nil {
		log.Printf("failed to create topics: %v", err)
	}
	err = stock.RegisterSchema()
	if err != nil {
		log.Printf("failed to register schema: %v", err)
	}
}
