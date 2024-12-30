<?php

namespace actor;

class NSFWInput
{
    # TODO: okの時priceがあり、extremeOkの時extremeSurchargeがあり、okじゃないときextremeOkもない等のvalidation
    public bool $ok;
    public int $price;
    public bool $extremeOk;
    public int $extremeSurcharge;
    public \common\Id $accountId;

    public function __construct(bool $ok, int $price, bool $hardOk, int $extremeSurcharge, \common\Id $accountId)
    {
        $this->ok = $ok;
        $this->price = $price;
        $this->extremeOk = $hardOk;
        $this->extremeSurcharge = $extremeSurcharge;
        $this->accountId = $accountId;
    }
}
