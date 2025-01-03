<?php

namespace koemade\auth;

use \koemade\auth\exceptions\InvalidRoleException;
use \koemade\auth\exceptions\InvalidStatusException;

class Claims
{
    public int $sub;    // ユーザーID
    public int $exp;    // 有効期限
    public int $iat;    // 発行時刻
    public int $nbf;    // 有効開始時刻
    public string $role;   // ユーザーのロール（例: "admin", "user"）
    public string $status; // ユーザーのステータス（例: "active", "banned", "suspended"）

    public function __construct(int $sub, int $exp, int $iat, int $nbf, string $role, string $status)
    {
        $validRoles = ['admin', 'actor', 'guest'];
        $validStatuses = ['active', 'banned', 'suspended'];

        if (!in_array($role, $validRoles)) {
            throw new InvalidRoleException("Invalid role: $role");
        }

        if (!in_array($status, $validStatuses)) {
            throw new InvalidStatusException("Invalid status: $status");
        }

        $this->sub = $sub;
        $this->exp = $exp;
        $this->iat = $iat;
        $this->nbf = $nbf;
        $this->role = $role;
        $this->status = $status;
    }
}
