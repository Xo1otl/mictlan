<?php

namespace submission;

class Address
{
    public string $text;

    public function __construct(string $text)
    {
        $this->text = $text;
    }

    function __toString(): string
    {
        return $this->text;
    }
}
