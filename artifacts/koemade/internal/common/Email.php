<?php

namespace common;

class Email
{
    public string $text;

    public function __construct(string $value)
    {
        if (!filter_var($value, FILTER_VALIDATE_EMAIL)) {
            throw new \InvalidArgumentException("Invalid email address");
        }
        $this->text = $value;
    }

    function __toString(): string
    {
        return $this->text;
    }
}
