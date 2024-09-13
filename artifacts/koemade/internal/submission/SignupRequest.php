<?php

namespace submission;

/**
 * 記録するアカウント作成申請情報
 *
 * 読み取り用のモデルだけどこの形で必要かわからないから編集する可能性もある
 */
class SignupRequest
{
    public \common\Id $id;
    public \common\Email $email;
    public JapaneseName $japaneseName;
    public Address $address;
    public Tel $tel;
    public \common\IdImage $idImage;
    public BeneficiaryInfo $beneficiaryInfo;
    public string $selfPromotion;

    public function __construct(
        \common\Id      $id,
        \common\Email   $email,
        JapaneseName    $japaneseName,
        Address         $address,
        Tel             $tel,
        \common\IdImage $idImage,
        BeneficiaryInfo $beneficiaryInfo,
        string          $selfPromotion
    )
    {
        $this->id = $id;
        $this->email = $email;
        $this->japaneseName = $japaneseName;
        $this->address = $address;
        $this->tel = $tel;
        $this->idImage = $idImage;
        $this->beneficiaryInfo = $beneficiaryInfo;

        if (strlen($selfPromotion) > 500) {
            throw new \InvalidArgumentException('selfPromotion must be 500 characters or less');
        }

        $this->selfPromotion = $selfPromotion;
    }
}
