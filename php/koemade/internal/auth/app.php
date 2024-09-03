<?php

namespace auth;

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
        try {
            $account = $this->accountRepo->findByUsername($signInInput->username);
        } catch (\Exception $e) {
            new \logger\Imp("account not found for username: " . $signInInput->username);
            return null;
        }
        if ($account->password->verify($signInInput->passwordText)) {
            $session = new Session($account->id, $account->username, $account->role);
            $this->sessionRepo->set($session);
        } else {
            new \logger\Imp("password is not correct");
            return null;
        }
        return $account;
    }

    public function deleteAccount(SignInInput $signInInput)
    {
        $account = $this->signin($signInInput);
        if ($account !== null) {
            $this->accountRepo->deleteById($account->id);
        }
    }

    public function deleteAccountByUsername(Username $username)
    {
        $this->accountRepo->deleteByUsername($username);
    }

    public function editPassword(EditPasswordInput $input)
    {
        new \logger\Debug("editing password for username: " . $input->username);
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