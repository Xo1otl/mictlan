package stock

import (
	"context"
	"time"
	"zaiko/internal/auth"

	"github.com/shopspring/decimal"
	"go.mongodb.org/mongo-driver/v2/bson"
	"go.mongodb.org/mongo-driver/v2/mongo"
	"go.mongodb.org/mongo-driver/v2/mongo/options"
)

type MongoRepo struct {
	client *mongo.Client
}

// Sales implements Repo.
func (m *MongoRepo) Sales(sub auth.Sub) decimal.Decimal {
	collection := m.client.Database("zaiko").Collection("sales_stocks")
	filter := bson.M{"Sub": string(sub)}

	// projectionでSalesフィールドだけを取得
	projection := bson.M{"Sales": 1}

	var result struct {
		Sales string `bson:"Sales"`
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// FindOneにprojectionを指定
	err := collection.FindOne(ctx, filter, options.FindOne().SetProjection(projection)).Decode(&result)
	if err != nil {
		return decimal.Zero
	}

	salesDecimal, err := decimal.NewFromString(result.Sales)
	if err != nil {
		return decimal.Zero
	}

	return salesDecimal
}

// Stocks implements Repo.
func (m *MongoRepo) Stocks(sub auth.Sub, name string) map[string]int {
	collection := m.client.Database("zaiko").Collection("sales_stocks")
	filter := bson.M{"Sub": string(sub)}

	// projectionでStocksフィールドだけを取得
	projection := bson.M{"Stocks": 1}

	var result struct {
		Stocks map[string]int `bson:"Stocks"`
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// FindOneにprojectionを指定
	err := collection.FindOne(ctx, filter, options.FindOne().SetProjection(projection)).Decode(&result)
	if err != nil {
		return nil
	}

	if name != "" {
		if value, exists := result.Stocks[name]; exists {
			return map[string]int{name: value}
		}
		return map[string]int{}
	}

	return result.Stocks
}

func NewMongoRepo() Repo {
	clientOptions := options.Client().ApplyURI("mongodb://mongo:27017")
	client, err := mongo.Connect(clientOptions)
	if err != nil {
		panic(err)
	}

	err = client.Ping(context.TODO(), nil)
	if err != nil {
		panic(err)
	}

	return &MongoRepo{client: client}
}
