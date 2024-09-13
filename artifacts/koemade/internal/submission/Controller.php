<?php

namespace submission;

class Controller
{
    private App $app;

    public function __construct(App $app)
    {
        $this->app = $app;
    }

    public function submitSignupRequest(array $postData, array $fileData, Storage $storage)
    {
        $idImageFile = $fileData['id_image'];
        if ($idImageFile['error'] !== UPLOAD_ERR_OK) {
            \logger\fatal("upload err not ok");
        }

        $createdAt = new \DateTime();
        $filename = pathinfo($idImageFile['tmp_name'], PATHINFO_FILENAME);
        $idImage = new \common\IdImage($filename, $idImageFile['type'], $idImageFile['size'], $createdAt);
        \logger\imp("upload id image from tmp", $idImage);
        $storage->uploadIdImageFromTmp($idImageFile['tmp_name'], $idImage);

        $japaneseName = new JapaneseName($postData['name'], $postData['furigana']);
        $address = new Address($postData['address']);
        $email = new \common\Email($postData['email']);
        $tel = new Tel($postData['tel']);
        $beneficiaryInfo = new BeneficiaryInfo($postData['bankName'], $postData['branchName'], $postData['accountNumber']);
        $selfPromotion = $postData['selfPromotion'];
        $signupRequestInput = new SignupRequestInput($email, $japaneseName, $address, $tel, $idImage, $beneficiaryInfo, $selfPromotion);

        $this->app->submitSignupRequest($signupRequestInput);
    }
}
