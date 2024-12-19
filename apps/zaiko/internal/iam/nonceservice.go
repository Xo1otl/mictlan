package iam

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// HMACNonceService は NonceService の実装です。
type HMACNonceService struct {
	serverSecret string
}

// NewHMACNonceService は NonceService を生成します。
func NewHMACNonceService(secret string) NonceService {
	return &HMACNonceService{
		serverSecret: secret,
	}
}

// Generate は nonce を生成します。
func (n *HMACNonceService) Generate() string {
	// 現在時刻のタイムスタンプを取得
	timestamp := strconv.FormatInt(time.Now().Unix(), 10)
	// タイムスタンプに基づき HMAC で署名を生成
	mac := hmac.New(sha256.New, []byte(n.serverSecret))
	mac.Write([]byte(timestamp))
	signature := base64.StdEncoding.EncodeToString(mac.Sum(nil))

	// nonce は "タイムスタンプ:署名" の形式で生成される
	nonce := fmt.Sprintf("%s:%s", timestamp, signature)
	return base64.StdEncoding.EncodeToString([]byte(nonce))
}

// Validate は nonce を検証します。
func (n *HMACNonceService) Validate(nonce string) error {
	// base64 デコードして nonce を取得
	decodedNonce, err := base64.StdEncoding.DecodeString(nonce)
	if err != nil {
		return errors.New("invalid nonce format")
	}

	// nonce を "タイムスタンプ:署名" 形式で分割
	parts := strings.Split(string(decodedNonce), ":")
	if len(parts) != 2 {
		return errors.New("invalid nonce format")
	}

	timestamp := parts[0]
	signature := parts[1]

	// タイムスタンプを元に再度署名を生成して検証
	mac := hmac.New(sha256.New, []byte(n.serverSecret))
	mac.Write([]byte(timestamp))
	expectedSignature := base64.StdEncoding.EncodeToString(mac.Sum(nil))

	if !hmac.Equal([]byte(signature), []byte(expectedSignature)) {
		return errors.New("invalid nonce signature")
	}

	// nonce の有効期限をチェック（例: 10分）
	nonceTime, err := strconv.ParseInt(timestamp, 10, 64)
	if err != nil {
		return errors.New("invalid nonce timestamp")
	}

	if time.Since(time.Unix(nonceTime, 0)) > 10*time.Minute {
		return errors.New("nonce has expired")
	}

	return nil
}
