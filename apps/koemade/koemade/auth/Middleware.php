<?php

namespace koemade\Internal\Auth;

class Middleware
{
    private $jwtService;

    public function __construct(JWTService $jwtService)
    {
        $this->jwtService = $jwtService;
    }

    public function verifyToken($token)
    {
        return $this->jwtService->verifyToken($token);
    }
}
