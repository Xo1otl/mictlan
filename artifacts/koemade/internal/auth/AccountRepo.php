<?php

namespace auth;

interface AccountRepo
{
    public function add(CredentialInput $credentialInput): \common\Id;

    public function editPassword(EditPasswordInput $input);

    public function deleteById(\common\Id $id);

    public function deleteByUsername(Username $username);

    /**
     * @throws \Exception
     */
    public function findByUsername(Username $username): Account;

    /**
     * @throws \Exception
     */
    public function findById(\common\Id $id): Account;

    /**
     * @return Account[]
     */
    public function getAllAccounts(): array;
}
