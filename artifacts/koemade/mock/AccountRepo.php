<?php

namespace mock;

use auth\EditPasswordInput;
use auth\Username;

class AccountRepo implements \auth\AccountRepo
{
    public function add(\auth\CredentialInput $credentialInput): \common\Id
    {
        return new \common\Id(1);
    }

    public function findByUsername(\common\Email $username): \auth\Account
    {
        $id = new \common\Id(1);
        $username = new \auth\Username("yamada@gmail.com");
        try {
            $password = "Abcd1234*";
        } catch (\Exception $e) {
            \logger\fatal($e);
        }
        $passwordHash = new \auth\PasswordHash(password_hash($password, PASSWORD_BCRYPT));
        return new \auth\Account($id, $username, $passwordHash);
    }

    public function findById(\common\Id $id): \auth\Account
    {
        \logger\fatal("not implemented");
    }

    public function deleteById(\common\Id $id)
    {
        // TODO: Implement delete() method.
    }

    public function deleteByUsername(Username $username)
    {
        // TODO: Implement deleteByUsername() method.
    }

    public function editPassword(EditPasswordInput $input)
    {
        // TODO: Implement editPassword() method.
    }

    public function getAllAccounts(): array
    {
        // TODO: Implement getAllAccounts() method.
        \logger\fatal("not implemented");
    }
}
