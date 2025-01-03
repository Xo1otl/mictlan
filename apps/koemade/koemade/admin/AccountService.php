<?php

namespace koemade\admin;

class CreateAccountInput
{
    public string $username;
    public string $password;

    public function __construct(string $username, string $password)
    {
        $this->validateUsername($username);
        $this->validatePassword($password);
        $this->username = $username;
        $this->password = $password;
    }

    private function validateUsername(string $username): void
    {
        if (empty($username)) {
            throw new \InvalidArgumentException("Username cannot be empty");
        }
        if (!filter_var($username, FILTER_VALIDATE_EMAIL)) {
            throw new \InvalidArgumentException("Username must be a valid email address");
        }
    }

    private function validatePassword(string $password): void
    {
        if (empty($password)) {
            throw new \InvalidArgumentException("Password cannot be empty");
        }
        if (strlen($password) < 6) {
            throw new \InvalidArgumentException("Password must be at least 6 characters long");
        }
    }
}

interface AccountService
{
    public function createAccount(CreateAccountInput $input): void;
    public function deleteAccount(string $username): void;
    public function banAccount(string $username): void;
}
