package iam

type CommandRepo interface {
	CreateAccount(username, password string) error
}
