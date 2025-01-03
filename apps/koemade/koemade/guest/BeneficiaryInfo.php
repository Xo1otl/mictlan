<?php

namespace koemade\guest;

class BeneficiaryInfo
{
    public string $bankName;
    public string $branchName;
    public string $accountNumber;

    public function __construct(string $bankName, string $branchName, string $accountNumber)
    {
        $this->bankName = $bankName;
        $this->branchName = $branchName;
        $this->accountNumber = $accountNumber;
    }
}