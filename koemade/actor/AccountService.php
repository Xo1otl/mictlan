<?php

namespace koemade\actor;

interface AccountService
{
    public function changePassword(string $account_id, string $oldPassword, string $newPassword): bool;
}
