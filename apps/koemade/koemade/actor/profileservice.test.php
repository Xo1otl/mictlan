<?php

require_once __DIR__ . "/../kernel/bootstrap.php";

use koemade\actor;
use koemade\dbadapter;

$profileService = new dbadapter\ProfileService();
$profileInput = new actor\ProfileInput(price: 1000, selfPromotion: "よろしくお願いします", extremeAllowed: true);
$profileService->save("10", $profileInput);
