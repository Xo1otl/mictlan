<?php

namespace auth;

class EditPasswordInput extends CredentialInput
{
    public Password $newPassword;

    public function __construct(Username $username, Password $password, Password $newPassword)
    {
        parent::__construct($username, $password);
        $this->newPassword = $newPassword;
    }
}
