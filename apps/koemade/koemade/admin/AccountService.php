<?php

namespace koemade\admin;

interface AccountService
{
    public function createAccount(CreateAccountInput $input): void;
    public function deleteAccount(string $username): void;
    public function banAccount(string $username): void;
}
