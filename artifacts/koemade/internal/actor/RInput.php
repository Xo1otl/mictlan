<?php

namespace actor;

class RInput
{
    # TODO: okの時priceがあり、hardOkの時hardSurchargeがあり、okじゃないときhardOkもない等のvalidation
    public bool $ok;
    public int $price;
    public bool $hardOk;
    public int $hardSurcharge;
    public \common\Id $accountId;

    public function __construct(bool $ok, int $price, bool $hardOk, int $hardSurcharge, \common\Id $accountId)
    {
        $this->ok = $ok;
        $this->price = $price;
        $this->hardOk = $hardOk;
        $this->hardSurcharge = $hardSurcharge;
        $this->accountId = $accountId;
    }
}
