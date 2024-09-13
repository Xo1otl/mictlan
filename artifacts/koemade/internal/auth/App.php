<?php

namespace auth;

/**
 * Authenticationアプリ
 *
 * アカウント情報は他のユーザーデータから独立しており、そのアカウント作成/削除とログイン/ログアウトとログインチェックができる
 */
class App
{
    private KeyService $keyService;
    private AccountRepo $accountRepo;
    private SessionRepo $sessionRepo;

    public function __construct(KeyService $keyService, AccountRepo $accountRepo, SessionRepo $sessionRepo)
    {
        $this->keyService = $keyService;
        $this->accountRepo = $accountRepo;
        $this->sessionRepo = $sessionRepo;
    }

    public function signup(CredentialInput $credentialInput): \common\Id
    {
        return $this->accountRepo->add($credentialInput);
    }

    public function signin(CredentialInput $credentialInput): ?Account
    {
        try {
            $account = $this->accountRepo->findByUsername($credentialInput->username);
        } catch (\Exception $e) {
            \logger\imp("user not found for username: ", $credentialInput->username);
            return null;
        }
        if ($this->keyService->passwordVerify($credentialInput->password, $account->passwordHash)) {
            $session = new Session($account->id, $account->username, $account->role);
            $this->sessionRepo->set($session);
        } else {
            \logger\imp("password is not correct");
            return null;
        }
        return $account;
    }

    public function deleteAccount(CredentialInput $credentialInput)
    {
        $account = $this->signin($credentialInput);
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
        \logger\debug($input);
        $this->accountRepo->editPassword($input);
    }

    /**
     * @return Account[]
     */
    public function getAllAccounts(): array
    {
        return $this->accountRepo->getAllAccounts();
    }
}
