package cognitojwt_test

import (
	"encoding/json"
	"log"
	"ossekaiserver/internal/auth"
	"ossekaiserver/internal/auth/cognitojwt"
	"testing"

	_ "embed"
)

//go:embed testcredentials.json
var testCredentialsJSON []byte

type Credentials struct {
	Token string `json:"token"`
}

func loadConfig() (Credentials, error) {
	var config Credentials
	err := json.Unmarshal(testCredentialsJSON, &config)
	return config, err
}

func TestTokenService(t *testing.T) {
	credentials, err := loadConfig()
	if err != nil {
		t.Fatal(err)
	}
	tokenService, err := cognitojwt.NewTokenService()
	if err != nil {
		t.Fatal(err)
	}
	token := auth.Token(credentials.Token)
	claims, err := auth.NewClaims(&token, tokenService.Parse)
	if err != nil {
		t.Fatal(err)
	}
	log.Println(claims)
}
