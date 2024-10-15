package auth

type TokenService interface {
	Parse(token *Token) (*Claims, error)
}
