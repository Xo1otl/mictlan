<?php

namespace auth;

class Password
{
    public string $text;

    public function __construct($text)
    {
        $this->validate($text);
        $this->text = $text;
    }

    private function validate(string $text): void
    {
        $lengthPattern = '/^.{8,}$/';
        $numberPattern = '/\d/';
        $uppercasePattern = '/[A-Z]/';
        $lowercasePattern = '/[a-z]/';
        $specialCharPattern = '/[!@#$%^&*(),.?":{}|<>]/';

        if (!preg_match($lengthPattern, $text)) {
            throw new \InvalidArgumentException('Password must be at least 8 characters long.');
        }
        if (!preg_match($numberPattern, $text)) {
            throw new \InvalidArgumentException('Password must contain at least one number.');
        }
        if (!preg_match($uppercasePattern, $text)) {
            throw new \InvalidArgumentException('Password must contain at least one uppercase letter.');
        }
        if (!preg_match($lowercasePattern, $text)) {
            throw new \InvalidArgumentException('Password must contain at least one lowercase letter.');
        }
        if (!preg_match($specialCharPattern, $text)) {
            throw new \InvalidArgumentException('Password must contain at least one special character.');
        }
    }
}
