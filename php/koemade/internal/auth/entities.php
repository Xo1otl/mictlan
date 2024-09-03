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
            throw new \InvalidArgumentException("invalid value: $value allowed values are: " . implode(', ', $roles));
        }
        $this->value = $value;
    }

    function __toString(): string
    {
        return $this->value;
    }
}

class Username extends \common\Email
{
    public function __construct(string $value)
    {
        parent::__construct($value);
    }
}


class Password
{
    private string $hash;

    public function __construct(string $hashOrPlainText, bool $needsHashing = true)
    {
        if ($needsHashing) {
            $this->setPassword($hashOrPlainText);
        } else {
            $this->setHashedPassword($hashOrPlainText);
        }
    }

    private function setPassword(string $plainText): void
    {
        $this->validate($plainText);
        $this->hash = password_hash($plainText, PASSWORD_BCRYPT);
    }

    private function setHashedPassword(string $hash): void
    {
        if (!$this->isValidHash($hash)) {
            throw new \InvalidArgumentException('invalid password hash format');
        }
        $this->hash = $hash;
    }

    private function validate(string $plainText): void
    {
        $lengthPattern = '/^.{8,}$/';
        $numberPattern = '/\d/';
        $uppercasePattern = '/[A-Z]/';
        $lowercasePattern = '/[a-z]/';
        $specialCharPattern = '/[!@#$%^&*(),.?":{}|<>]/';

        if (!preg_match($lengthPattern, $plainText)) {
            throw new \InvalidArgumentException('password must be at least 8 characters long');
        }
        if (!preg_match($numberPattern, $plainText)) {
            throw new \InvalidArgumentException('password must contain at least one number');
        }
        if (!preg_match($uppercasePattern, $plainText)) {
            throw new \InvalidArgumentException('password must contain at least one uppercase letter');
        }
        if (!preg_match($lowercasePattern, $plainText)) {
            throw new \InvalidArgumentException('password must contain at least one lowercase letter');
        }
        if (!preg_match($specialCharPattern, $plainText)) {
            throw new \InvalidArgumentException('password must contain at least one special character');
        }
    }

    private function isValidHash(string $hash): bool
    {
        return password_get_info($hash)['algo'] !== null;
    }

    public function verify(string $plainText): bool
    {
        return password_verify($plainText, $this->hash);
    }

    public function __toString(): string
    {
        return $this->hash;
    }
}

class AccountId
{
    private string $id;
    public function __construct(string $id)
    {
        $this->id = $id;
    }
    public function __toString(): string
    {
        return $this->id;
    }
}

class Account
{
    public AccountId $id;
    public Role $role;
    public Username $username;
    public Password $password;

    public function __construct(AccountId $id, Username $username, Password $password, ?Role $role = null)
    {
        $this->id = $id;
        $this->role = $role ?? new Role(Role::ACTOR);
        $this->username = $username;
        $this->password = $password;
    }
}

class Session
{
    public AccountId $accountId;
    public Username $username;
    public Role $role;
    public \DateTimeImmutable $createdAt;
    public \DateTimeImmutable $expiresAt;

    public function __construct(
        AccountId $accountId,
        Username $username,
        Role $role,
        \DateTimeImmutable $expiresAt = null
    ) {
        $this->accountId = $accountId;
        $this->username = $username;
        $this->role = $role;
        $this->createdAt = new \DateTimeImmutable();
        $this->expiresAt = $expiresAt ?? $this->createdAt->modify('+1 hour');
    }
}

class SignUpInput
{
    public Username $username;
    public Password $password;
    public function __construct(Username $username, Password $password)
    {
        $this->username = $username;
        $this->password = $password;
    }
}

class SignInInput
{
    public Username $username;
    public string $passwordText;
    public function __construct(Username $username, string $passwordText)
    {
        $this->username = $username;
        $this->passwordText = $passwordText;
    }
}

class EditPasswordInput extends SignInInput
{
    public Password $newPassword;
    public function __construct(Username $username, Password $password, Password $newPassword)
    {
        parent::__construct($username, $password);
        $this->newPassword = $newPassword;
    }
}