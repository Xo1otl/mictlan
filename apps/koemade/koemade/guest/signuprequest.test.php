<?php

require_once __DIR__ . '/../kernel/bootstrap.php';

use koemade\guest;

$japaneseName = new guest\JapaneseName("山田太郎", "ヤマダタロウ");
$beneficiaryInfo = new guest\BeneficiaryInfo("東京銀行", "新宿支店", "1234567");

$signupRequest = new guest\SignupRequest(
    "taro.yamada@example.com",
    $japaneseName,
    "東京都新宿区",
    "03-1234-5678",
    "/tmp/demo.png",
    $beneficiaryInfo,
    "よろしくお願いします。"
);

$service = new guest\MailService();
$service->notify($signupRequest);
