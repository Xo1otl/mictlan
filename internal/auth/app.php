<?php

namespace koemade\auth;

class App
{
    private AccountRepo $accountRepo;
    private SessionRepo $sessionRepo;

    public function __construct(AccountRepo $accountRepo, SessionRepo $sessionRepo)
    {
        $this->accountRepo = $accountRepo;
        $this->sessionRepo = $sessionRepo;
    }

    public function signup(SignUpInput $signUpInput): AccountId
    {
        return $this->accountRepo->add($signUpInput);
    }

    public function signin(SignInInput $signInInput): ?Account
    {
        $account = $this->accountRepo->findByUsername($signInInput->username);
        if (!$account->password->verify($signInInput->passwordText)) {
            throw new \Exception("password is not correct");
        }
        $session = new Session($account->id, $account->username, $account->role);
        $this->sessionRepo->set($session);
        return $account;
    }

    public function deleteAccount(SignInInput $signInInput)
    {
        $account = $this->accountRepo->findByUsername($signInInput->username);
        if (!$account->password->verify($signInInput->passwordText)) {
            throw new \Exception("password is not correct");
        }
        $this->accountRepo->deleteById($account->id);
    }

    public function deleteAccountByUsername(Username $username)
    {
        $this->accountRepo->deleteByUsername($username);
    }

    public function editPassword(EditPasswordInput $input)
    {
        $this->accountRepo->editPassword($input);
    }

    /**
     * @return Account[]
     */
    public function allAccounts(): array
    {
        return $this->accountRepo->allAccounts();
    }
}
