package auth

type Token string

type Sub string

type Claims struct {
	Sub
}

func NewClaims(token *Token, parse ParseToken) (*Claims, error) {
	return parse(token)
}
