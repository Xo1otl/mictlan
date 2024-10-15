package auth

type Token string

type Sub string

type Claims struct {
	Sub
}

func NewClaims(token *Token, service TokenService) (*Claims, error) {
	return service.Parse(token)
}
