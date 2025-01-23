<?php

require_once __DIR__ . "/../kernel/bootstrap.php";

use koemade\dbadapter;

$accountService = new dbadapter\AccountService();
$accountService->changePassword("2", "Abcd1234*", "newpassword");
$accountService->changePassword("2", "newpassword", "Abcd1234*");
$accountService->changePassword("2", "wrongpassword", "Abcd1234*");
