<?php

namespace bcrypt;

class KeyService implements \auth\KeyService
{
    public function passwordVerify(\auth\Password $password, \auth\PasswordHash $passwordHash): bool
    {
        return password_verify($password->text, $passwordHash->text);
    }
}
