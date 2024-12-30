<?php

namespace actor;

class NSFWOptions
{
    public bool $ok;
    public int $price;
    public bool $extremeOk;
    public int $extremeSurcharge;

    public function __construct(bool $ok, int $price, bool $extremeOk, int $extremeSurcharge)
    {
        $this->ok = $ok;
        $this->price = $price;
        $this->extremeOk = $extremeOk;
        $this->extremeSurcharge = $extremeSurcharge;
    }
}
