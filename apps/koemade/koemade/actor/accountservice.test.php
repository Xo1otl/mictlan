<?php

require_once __DIR__ . "/../kernel/bootstrap.php";

use koemade\dbadapter;

$accountService = new dbadapter\AccountService();
$accountService->changePassword("qlovolp.ttt@gmail.com", "Abcd1234*", "newpassword");
$accountService->changePassword("qlovolp.ttt@gmail.com", "newpassword", "Abcd1234*");
$accountService->changePassword("qlovolp.ttt@gmail.com", "wrongpassword", "Abcd1234*");
