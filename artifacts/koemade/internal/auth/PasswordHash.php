<?php

namespace auth;

class PasswordHash
{
    public string $text;

    public function __construct($text)
    {
        $this->text = $text;
    }
}
