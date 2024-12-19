package iam

import (
	"fmt"
	"log"
	"strconv"
	"strings"
	"zaiko/internal/auth"
)

type DigestToken struct {
	Username string
	Realm    string
	Nonce    string
	URI      string
	Cnonce   string
	Nc       string
	Qop      string
	Response string
	Opaque   string
}

// NewDigestTokenはstring形式のトークンを受け取り、DigestTokenを生成します
func NewDigestToken(authToken *auth.Token) (DigestToken, error) {
	// "Digest username=\"john\", realm=\"localhost\", ..."のような形式を期待
	if !strings.HasPrefix(string(*authToken), "Digest ") {
		return DigestToken{}, fmt.Errorf("invalid token format")
	}
	// "Digest "を除去し、コンマで分割
	tokenParts := strings.Split(string(*authToken)[6:], ", ")
	digestToken := DigestToken{}
	// 各部分を解析
	for _, part := range tokenParts {
		// "="で分割
		kv := strings.SplitN(part, "=", 2)
		if len(kv) != 2 {
			return DigestToken{}, fmt.Errorf("invalid token part: %s", part)
		}
		key := strings.TrimSpace(kv[0])
		value := strings.Trim(kv[1], "\"")
		// フィールドに値をセット
		switch key {
		case "username":
			digestToken.Username = value
		case "realm":
			digestToken.Realm = value
		case "nonce":
			digestToken.Nonce = value
		case "uri":
			digestToken.URI = value
		case "cnonce":
			digestToken.Cnonce = value
		case "nc":
			digestToken.Nc = value
		case "qop":
			digestToken.Qop = value
		case "response":
			digestToken.Response = value
		case "opaque":
			digestToken.Opaque = value
		default:
			log.Println("Unknown key: ", key)
		}
	}
	return digestToken, nil
}

type Digest struct {
	accountRepo  AccountRepo
	ncRepo       DigestNcRepo
	nonceService NonceService
	DigestTokenValidator
}

func NewDigest(queryRepo AccountRepo, ncRepo DigestNcRepo, nonceService NonceService, validator DigestTokenValidator) *Digest {
	return &Digest{accountRepo: queryRepo, ncRepo: ncRepo, nonceService: nonceService, DigestTokenValidator: validator}
}

func (d *Digest) Init(realm string) DigestToken {
	nonce := d.nonceService.Generate()
	d.ncRepo.set(nonce, "00000001")
	return DigestToken{
		Nonce:  nonce,
		Opaque: "opaque",
		Qop:    "auth",
		Realm:  realm,
	}
}

func (d *Digest) Parse(token *auth.Token) (*auth.Claims, error) {
	// トークンをパースしてDigestTokenを生成
	digestToken, err := NewDigestToken(token)
	if err != nil {
		return nil, err
	}
	log.Println("Digest token parsed successfully")

	// nonceを検証
	err = d.nonceService.Validate(digestToken.Nonce)
	if err != nil {
		return nil, err
	}
	log.Println("Nonce validated successfully")

	// 現在のnonce count (nc) を取得
	currentNc := d.ncRepo.get(digestToken.Nonce)

	// クライアントから送られてきたncと比較し、一致しない場合はエラー
	if digestToken.Nc != currentNc {
		return nil, fmt.Errorf("invalid nonce count")
	}
	log.Println("Nonce count validated successfully")

	// アカウントを取得
	account, err := d.accountRepo.FindAccount(digestToken.Username)
	if err != nil {
		return nil, err
	}

	// トークンを検証
	if err := d.DigestTokenValidator.Validate(digestToken, account.PasswordHash); err != nil {
		return nil, err
	}

	// nonce countをインクリメント
	newNc, err := incrementNonceCount(currentNc)
	if err != nil {
		return nil, fmt.Errorf("failed to increment nonce count: %v", err)
	}

	// 新しいncを保存
	if err := d.ncRepo.set(digestToken.Nonce, newNc); err != nil {
		return nil, fmt.Errorf("failed to update nonce count: %v", err)
	}

	log.Println("All validations succeeded")
	// すべて成功した場合はClaimsを返す
	return &auth.Claims{Sub: auth.Sub(digestToken.Username)}, nil
}

// incrementNonceCountはncのカウントをインクリメントする関数です。
// ncは16進数形式の値として保存されている前提です。
func incrementNonceCount(currentNc string) (string, error) {
	// 16進数形式のncをintに変換
	ncInt, err := strconv.ParseUint(currentNc, 16, 64)
	if err != nil {
		return "", err
	}

	// ncをインクリメント
	ncInt++

	// 再び16進数に変換し、ゼロ埋めして返す（例えば、"00000001"の形式）
	return fmt.Sprintf("%08x", ncInt), err
}

type DigestNcRepo interface {
	set(nonce, nc string) error
	get(nonce string) string
}

type NonceService interface {
	Generate() string
	Validate(nonce string) error
}

type DigestTokenValidator interface {
	// Validateはクライアントから送られてきたDigestTokenとpasswordHashを検証する
	Validate(token DigestToken, passwordHash string) error
}
