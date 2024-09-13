<?php

namespace auth;

class CredentialInput
{
    public Username $username;
    public Password $password;

    public function __construct(Username $username, Password $password)
    {
        $this->username = $username;
        $this->password = $password;
    }
}
