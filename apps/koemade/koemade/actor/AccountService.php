<?php

namespace koemade\actor;

interface AccountService
{
    public function changePassword(string $username, string $oldPassword, string $newPassword): bool;
}
