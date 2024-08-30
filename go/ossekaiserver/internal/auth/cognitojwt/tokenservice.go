package cognitojwt

import (
	"context"
	"encoding/json"
	"fmt"
	"ossekaiserver/internal/auth"
	"sync"
	"time"

	"github.com/lestrrat-go/jwx/v2/jwk"
	"github.com/lestrrat-go/jwx/v2/jwt"

	_ "embed"
)

//go:embed awsconfig.json
var configJSON []byte

type Config struct {
	AwsCognitoRegion string `json:"aws_cognito_region"`
	AwsUserPoolsID   string `json:"aws_user_pools_id"`
}

func loadConfig() (Config, error) {
	var config Config
	err := json.Unmarshal(configJSON, &config)
	return config, err
}

type TokenService struct {
	keySet jwk.Set
	issuer string
	mu     sync.RWMutex
}

func NewTokenService() (*TokenService, error) {
	config, err := loadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %v", err)
	}

	keyURL := fmt.Sprintf("https://cognito-idp.%s.amazonaws.com/%s/.well-known/jwks.json", config.AwsCognitoRegion, config.AwsUserPoolsID)
	keySet, err := jwk.Fetch(context.Background(), keyURL)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch JWK set: %v", err)
	}

	issuer := fmt.Sprintf("https://cognito-idp.%s.amazonaws.com/%s", config.AwsCognitoRegion, config.AwsUserPoolsID)

	return &TokenService{
		keySet: keySet,
		issuer: issuer,
	}, nil
}

func (v *TokenService) Parse(token *auth.Token) (*auth.Claims, error) {
	v.mu.RLock()
	defer v.mu.RUnlock()

	parsed, err := jwt.Parse(
		[]byte(*token),
		jwt.WithKeySet(v.keySet),
		jwt.WithValidate(true),
		jwt.WithIssuer(v.issuer),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to parse JWT: %v", err)
	}

	if time.Now().After(parsed.Expiration()) {
		return nil, fmt.Errorf("token has expired")
	}

	sub := parsed.Subject()
	claims := &auth.Claims{
		Sub: auth.Sub(sub),
	}
	return claims, nil
}

// 定期的にキーセットを更新するメソッド（必要に応じて使用）
func (v *TokenService) RefreshKeySet() error {
	config, err := loadConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %v", err)
	}

	keyURL := fmt.Sprintf("https://cognito-idp.%s.amazonaws.com/%s/.well-known/jwks.json", config.AwsCognitoRegion, config.AwsUserPoolsID)
	newKeySet, err := jwk.Fetch(context.TODO(), keyURL)
	if err != nil {
		return fmt.Errorf("failed to fetch JWK set: %v", err)
	}

	v.mu.Lock()
	v.keySet = newKeySet
	v.mu.Unlock()

	return nil
}
