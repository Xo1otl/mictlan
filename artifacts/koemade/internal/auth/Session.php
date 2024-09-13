<?php

namespace auth;

class Session
{
    public \common\Id $accountId;
    public Username $username;
    public Role $role;
    public \DateTimeImmutable $createdAt;
    public \DateTimeImmutable $expiresAt;

    public function __construct(
        \common\Id         $accountId,
        Username           $username,
        Role               $role,
        \DateTimeImmutable $expiresAt = null
    )
    {
        $this->accountId = $accountId;
        $this->username = $username;
        $this->role = $role;
        $this->createdAt = new \DateTimeImmutable();
        $this->expiresAt = $expiresAt ?? $this->createdAt->modify('+1 hour');
    }
}
