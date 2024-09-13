<?php

namespace common;

class Id
{
    public int $value;

    public function __construct(int $value)
    {
        if ($value <= 0) {
            throw new \InvalidArgumentException("Invalid ID value");
        }
        $this->value = $value;
    }

    function __toInt(): int
    {
        return $this->value;
    }

    function __toString(): string
    {
        return $this->value;
    }
}
