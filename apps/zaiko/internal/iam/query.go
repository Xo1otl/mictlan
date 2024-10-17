package iam

type AccountRepo interface {
	FindAccount(username string) (*Account, error)
}

type Account struct {
	Id           AccountId
	Username     string
	PasswordHash string
}

func NewAccount(id AccountId, username, passwordHash string) *Account {
	return &Account{
		Id:           id,
		Username:     username,
		PasswordHash: passwordHash,
	}
}

type AccountId string
