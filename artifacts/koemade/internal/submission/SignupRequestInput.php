<?php

namespace submission;

/**
 * アカウント作成申請で提出する情報
 *
 * Idはリポジトリが作成してくれるため入力データには含まれない
 * というかユーザーから渡されてくるデータと実際に持ちたいデータの形が一緒である保証は全くないので今後のためにもInputとObjectは分ける
 */
class SignupRequestInput
{
    public \common\Email $email;
    public JapaneseName $japaneseName;
    public Address $address;
    public Tel $tel;
    public \common\IdImage $idImage;
    public BeneficiaryInfo $beneficiaryInfo;
    public string $selfPromotion;

    public function __construct(
        \common\Email   $email,
        JapaneseName    $japaneseName,
        Address         $address,
        Tel             $tel,
        \common\IdImage $idImage,
        BeneficiaryInfo $beneficiaryInfo,
        string          $selfPromotion
    )
    {
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
