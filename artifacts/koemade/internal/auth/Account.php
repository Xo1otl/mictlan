<?php

namespace auth;

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
