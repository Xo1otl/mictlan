<?php

require_once __DIR__ . "/../kernel/bootstrap.php";

use koemade\actor;
use koemade\dbadapter;

$profileService = new dbadapter\ProfileService();
$profileInput = new actor\ProfileInput(
    "山田太郎",
    "よろしくお願いします",
    1000,
    null,
    null,
    null,
    null,
    5500
);
$profileService->save("10", $profileInput);
