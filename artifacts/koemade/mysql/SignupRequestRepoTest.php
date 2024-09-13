<?php

namespace mysql;

require __DIR__ . '/../vendor/autoload.php';

function generateRandomSignupRequest(): \submission\SignupRequestInput
{

    $createdAt = new \DateTime();
    $idImage = new \common\IdImage("demo", "image/jpeg", 200, $createdAt);
    $japaneseName = new \submission\JapaneseName("山田", "やまだ");
    $address = new \submission\Address("東京");
    $email = new \common\Email("yamada@koemade.net");
    $tel = new \submission\Tel("09012345678");
    $beneficiaryInfo = new \submission\BeneficiaryInfo("三菱東京UFJ銀行", "野並支店", "1234-1234-1234-1234");
    return new \submission\SignupRequestInput($email, $japaneseName, $address, $tel, $idImage, $beneficiaryInfo, "ABC");
}

function TestSignupRequestRepo()
{
    $db = new SignupRequestRepo();
    $info = generateRandomSignupRequest();
    $db->add($info);
}

TestSignupRequestRepo();
