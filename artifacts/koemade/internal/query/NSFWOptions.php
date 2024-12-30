<?php

namespace query;

class NSFWOptions
{
    public function __construct(bool $ok, bool $extremeOk)
    {
        $this->ok = $ok;
        $this->extremeOk = $extremeOk;
    }
}
