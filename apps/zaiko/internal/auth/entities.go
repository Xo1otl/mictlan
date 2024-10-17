package auth

type Token string

type Sub string

type Claims struct {
	Sub
}

func NewClaims(token *Token, tokenService TokenService) (*Claims, error) {
	return tokenService.Parse(token)
}
