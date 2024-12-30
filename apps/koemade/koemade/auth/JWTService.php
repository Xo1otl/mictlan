<?php

namespace koemade\Internal\Auth;

interface JWTService
{
    public function verifyToken($token): array;
}
