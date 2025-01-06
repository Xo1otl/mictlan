<?php

namespace koemade\admin;

class CreateAccountInput
{
    // ロールの定数
    public const ROLE_ACTOR = 'actor';
    public const ROLE_ADMIN = 'admin';
    public const ROLE_GUEST = 'guest';

    // 有効なロールのリスト
    private const VALID_ROLES = [
        self::ROLE_ACTOR,
        self::ROLE_ADMIN,
        self::ROLE_GUEST,
    ];

    public string $username;
    public string $password;
    public array $roles;

    public function __construct(string $username, string $password, array $roles = [self::ROLE_ACTOR])
    {
        $this->validateUsername($username);
        $this->validatePassword($password);
        $this->validateRoles($roles);

        $this->username = $username;
        $this->password = $password;
        $this->roles = $roles;
    }

    private function validateUsername(string $username): void
    {
        if (empty($username)) {
            throw new \InvalidArgumentException("Username cannot be empty");
        }
        if (!filter_var($username, FILTER_VALIDATE_EMAIL)) {
            throw new \InvalidArgumentException("Username must be a valid email address");
        }
    }

    private function validatePassword(string $password): void
    {
        if (empty($password)) {
            throw new \InvalidArgumentException("Password cannot be empty");
        }
        if (strlen($password) < 6) {
            throw new \InvalidArgumentException("Password must be at least 6 characters long");
        }
    }

    private function validateRoles(array $roles): void
    {
        if (empty($roles)) {
            throw new \InvalidArgumentException("Roles cannot be empty");
        }

        foreach ($roles as $role) {
            if (!in_array($role, self::VALID_ROLES, true)) {
                throw new \InvalidArgumentException("Invalid role: $role");
            }
        }
    }
}
