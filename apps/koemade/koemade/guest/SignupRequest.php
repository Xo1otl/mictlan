<?php

namespace koemade\guest;


class SignupRequest
{
    public string $email;
    public JapaneseName $japaneseName;
    public string $address;
    public string $tel;
    public string $idImagePath;
    public BeneficiaryInfo $beneficiaryInfo;
    public string $selfPromotion;

    public function __construct(
        string $email,
        JapaneseName $japaneseName,
        string $address,
        string $tel,
        string $idImagePath,
        BeneficiaryInfo $beneficiaryInfo,
        string $selfPromotion
    ) {
        $this->email = $email;
        $this->japaneseName = $japaneseName;
        $this->address = $address;
        $this->tel = $tel;
        $this->idImagePath = $idImagePath;
        $this->beneficiaryInfo = $beneficiaryInfo;

        if (strlen($selfPromotion) > 500) {
            throw new \InvalidArgumentException('selfPromotion must be 500 characters or less');
        }

        $this->selfPromotion = $selfPromotion;
    }
}
