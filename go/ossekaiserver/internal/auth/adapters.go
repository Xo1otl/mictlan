package auth

type ParseToken func(token *Token) (*Claims, error)
