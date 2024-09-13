<?php

namespace auth;

class Controller
{
    private App $app;

    public function __construct(App $app)
    {
        $this->app = $app;
    }

    public function handleSignup(array $postData): \common\Id
    {
        try {
            $username = new Username($postData['email']);
            $password = new Password($postData['password']);
            $credential = new CredentialInput($username, $password);
            return $this->app->signup($credential);
        } catch (\Exception $e) {
            throw new \RuntimeException($e->getMessage());
        }
    }

    public function handleSignin(array $postData)
    {
        try {
            $username = new Username($postData['username']);
            $password = new Password($postData['password']);
            $credential = new CredentialInput($username, $password);
            $this->app->signin($credential);
        } catch (\Exception $e) {
            \logger\imp($e);
            return;
        }
    }

    public function handleDeleteAccount(array $postData)
    {
        try {
            $username = new Username($postData['username']);
            $password = new Password($postData['password']);
            $credential = new CredentialInput($username, $password);
            $this->app->deleteAccount($credential);
        } catch (\Exception $e) {
            \logger\imp($e);
            return;
        }
    }

    public function handleDeleteAccountByAdmin(array $postData)
    {
        $username = new Username($postData['username']);
        $this->app->deleteAccountByUsername($username);
    }

    public function handleEditPassword(array $postData)
    {
        $username = new Username($postData['username']);
        $oldPassword = new Password($postData['old-password']);
        $newPassword = new Password($postData['new-password']);
        $input = new EditPasswordInput($username, $oldPassword, $newPassword);
        $this->app->editPassword($input);
    }

    /**
     * @return Account[]
     */
    public function handleGetAllAccounts(): array
    {
        return $this->app->getAllAccounts();
    }
}
