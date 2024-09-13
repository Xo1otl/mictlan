<?php

namespace actor;

class R
{
    public bool $ok;
    public int $price;
    public bool $hardOk;
    public int $hardSurcharge;

    public function __construct(bool $ok, int $price, bool $hardOk, int $hardSurcharge)
    {
        $this->ok = $ok;
        $this->price = $price;
        $this->hardOk = $hardOk;
        $this->hardSurcharge = $hardSurcharge;
    }
}
