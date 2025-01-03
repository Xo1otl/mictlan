<?php

namespace koemade\admin;

interface AdminService
{
    public function loginAsAccount(string $username): string;
}
