<?php

require_once __DIR__ . "/../kernel/bootstrap.php";

use koemade\actor;
use koemade\dbadapter;

$profileService = new dbadapter\ProfileService();
$profileInput = new actor\ProfileInput(displayName: "山田太郎", price: 1000, selfPromotion: "よろしくお願いします", extremeSurcharge: 5500);
$profileService->save("10", $profileInput);
