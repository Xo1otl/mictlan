<?php

namespace submission;

class Tel
{
    public string $number;
    private const PATTERN = '/^(?:\d{2,4}-\d{2,4}-\d{4}|\d{10,11})$/';

    public function __construct(string $number)
    {
        if (!preg_match(self::PATTERN, $number)) {
            throw new \InvalidArgumentException("Invalid telephone number");
        }
        $this->number = $number;
    }
}
