package auth_test

import (
	"encoding/json"
	"log"
	"ossekaiserver/internal/auth"
	"testing"

	_ "embed"
)

//go:embed testcredentials.json
var testCredentialsJSON []byte

type Credentials struct {
	Token string `json:"token"`
}

func loadCredentials() (Credentials, error) {
	var config Credentials
	err := json.Unmarshal(testCredentialsJSON, &config)
	return config, err
}

func TestAwsTokenService(t *testing.T) {
	credentials, err := loadCredentials()
	if err != nil {
		t.Fatal(err)
	}
	tokenService, err := auth.NewAwsTokenService()
	if err != nil {
		t.Fatal(err)
	}
	token := auth.Token(credentials.Token)
	claims, err := auth.NewClaims(&token, tokenService.Parse)
	if err == nil {
		t.Fatal("Expected error, got nil")
	}
	log.Println(claims)
}
