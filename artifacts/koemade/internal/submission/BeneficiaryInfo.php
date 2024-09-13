<?php

namespace submission;

/**
 * 口座情報
 *
 * TODO: Validationを追加する
 *  日本の口座だけを扱う場合かんたんなので仕様を聞く
 */
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
