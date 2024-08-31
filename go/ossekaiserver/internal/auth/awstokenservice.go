package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/lestrrat-go/jwx/v2/jwk"
	"github.com/lestrrat-go/jwx/v2/jwt"

	_ "embed"
)

//go:embed awsconfig.json
var awsConfigJSON []byte

type AwsConfig struct {
	AwsCognitoRegion string `json:"aws_cognito_region"`
	AwsUserPoolsID   string `json:"aws_user_pools_id"`
}

func loadAwsConfig() (AwsConfig, error) {
	var config AwsConfig
	err := json.Unmarshal(awsConfigJSON, &config)
	return config, err
}

type AwsTokenService struct {
	keySet jwk.Set
	issuer string
	mu     sync.RWMutex
}

func NewAwsTokenService() (*AwsTokenService, error) {
	config, err := loadAwsConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %v", err)
	}

	keyURL := fmt.Sprintf("https://cognito-idp.%s.amazonaws.com/%s/.well-known/jwks.json", config.AwsCognitoRegion, config.AwsUserPoolsID)
	keySet, err := jwk.Fetch(context.Background(), keyURL)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch JWK set: %v", err)
	}

	issuer := fmt.Sprintf("https://cognito-idp.%s.amazonaws.com/%s", config.AwsCognitoRegion, config.AwsUserPoolsID)

	return &AwsTokenService{
		keySet: keySet,
		issuer: issuer,
	}, nil
}

func (a *AwsTokenService) Parse(token *Token) (*Claims, error) {
	a.mu.RLock()
	defer a.mu.RUnlock()

	parsed, err := jwt.Parse(
		[]byte(*token),
		jwt.WithKeySet(a.keySet),
		jwt.WithValidate(true),
		jwt.WithIssuer(a.issuer),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to parse JWT: %v", err)
	}

	if time.Now().After(parsed.Expiration()) {
		return nil, fmt.Errorf("token has expired")
	}

	sub := parsed.Subject()
	claims := &Claims{
		Sub: Sub(sub),
	}
	return claims, nil
}

// 定期的にキーセットを更新するメソッド（必要に応じて使用）
func (a *AwsTokenService) RefreshKeySet() error {
	config, err := loadAwsConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %v", err)
	}

	keyURL := fmt.Sprintf("https://cognito-idp.%s.amazonaws.com/%s/.well-known/jwks.json", config.AwsCognitoRegion, config.AwsUserPoolsID)
	newKeySet, err := jwk.Fetch(context.TODO(), keyURL)
	if err != nil {
		return fmt.Errorf("failed to fetch JWK set: %v", err)
	}

	a.mu.Lock()
	a.keySet = newKeySet
	a.mu.Unlock()

	return nil
}
