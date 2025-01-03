<?php

namespace koemade\auth;

interface TokenService
{
    public function verify(string $token): Claims;
}
