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

namespace auth;

class Username extends \common\Email
{
}

class Account
{
    public \common\Id $id;
    public Role $role;
    public Username $username;
    public PasswordHash $passwordHash;

    public function __construct(\common\Id $id, Username $username, PasswordHash $passwordHash, ?Role $role = null)
    {
        $this->id = $id;
        $this->role = $role ?? new Role(Role::ACTOR);
        $this->username = $username;
        $this->passwordHash = $passwordHash;
    }
}

class PasswordHash
{
    public string $text;

    public function __construct($text)
    {
        $this->text = $text;
    }
}