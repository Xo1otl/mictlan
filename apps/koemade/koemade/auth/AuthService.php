<?php

namespace koemade\auth;

interface AuthService
{
    public function authenticate(string $username, string $password): string;
}
