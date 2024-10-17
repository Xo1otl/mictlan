package iam

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"log"
)

// MD5DigestValidator implements DigestTokenValidator interface
type MD5DigestValidator struct{}

func NewMD5DigestValidator() DigestTokenValidator {
	return &MD5DigestValidator{}
}

// Validate checks if the response hash from the client matches the one generated on the server side
func (v *MD5DigestValidator) Validate(token DigestToken, passwordHash string) error {
	// HA1 = MD5(username:realm:passwordHash)
	ha1 := computeMD5(fmt.Sprintf("%s:%s:%s", token.Username, token.Realm, passwordHash))
	log.Printf("HA1: %s (username: %s, realm: %s, passwordHash: %s)\n", ha1, token.Username, token.Realm, passwordHash)

	// HA2 = MD5(method:uri) -- assuming a fixed method (e.g., GET)
	// In a real implementation, the method should be part of the token.
	method := "GET" // Method should ideally come from token or context
	ha2 := computeMD5(fmt.Sprintf("%s:%s", method, token.URI))
	log.Printf("HA2: %s (method: %s, uri: %s)\n", ha2, method, token.URI)

	// Response = MD5(HA1:nonce:nc:cnonce:qop:HA2)
	expectedResponse := computeMD5(fmt.Sprintf("%s:%s:%s:%s:%s:%s", ha1, token.Nonce, token.Nc, token.Cnonce, token.Qop, ha2))
	log.Printf("Expected Response: %s (HA1: %s, nonce: %s, nc: %s, cnonce: %s, qop: %s, HA2: %s)\n",
		expectedResponse, ha1, token.Nonce, token.Nc, token.Cnonce, token.Qop, ha2)

	// Compare the calculated response with the one provided by the client
	log.Printf("Client Response: %s\n", token.Response)
	if expectedResponse != token.Response {
		log.Println("Digest token response validation failed")
		return fmt.Errorf("invalid digest token response")
	}

	log.Println("Digest token response validation succeeded")
	return nil
}

// computeMD5 is a helper function to calculate the MD5 hash
func computeMD5(data string) string {
	hash := md5.Sum([]byte(data))
	return hex.EncodeToString(hash[:])
}
