<?php

namespace auth;

class Role
{
    public string $value;

    public const ADMIN = 'admin';
    public const ACTOR = 'actor';
    public const GUEST = 'guest';

    public function __construct(string $value)
    {
        $roles = [self::ADMIN, self::ACTOR, self::GUEST];
        if (!in_array($value, $roles, true)) {
            throw new \InvalidArgumentException("Invalid value: $value. Allowed values are: " . implode(', ', $roles));
        }
        $this->value = $value;
    }

    function __toString(): string
    {
        return $this->value;
    }
}