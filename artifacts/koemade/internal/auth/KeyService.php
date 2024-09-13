<?php

namespace auth;

interface KeyService
{
    public function passwordVerify(Password $password, PasswordHash $passwordHash): bool;
}
