<?php

namespace koemade\auth;

interface AccountRepo
{
    public function add(SignUpInput $addAccountInput): AccountId;
    public function editPassword(EditPasswordInput $input);
    public function deleteById(AccountId $id);
    public function deleteByUsername(Username $username);
    /**
     * @throws \Exception
     */
    public function findByUsername(Username $username): Account;
    /**
     * @throws \Exception
     */
    public function findById(AccountId $id): Account;
    /**
     * @return Account[]
     */
    public function allAccounts(): array;
}

interface SessionRepo
{
    public function get(): ?Session;
    public function set(Session $session);
    public function delete();
}
