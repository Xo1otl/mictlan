<?php

require __DIR__ . '/../../vendor/autoload.php';

use koemade\auth;
use util\logger;

$repo = new auth\FileSessionRepo();
$accountId = new auth\AccountId(10);
$username = new auth\Username("abcd@gmail.com");
$role = new auth\Role("actor");
$session = new auth\Session($accountId, $username, $role);
$repo->set($session);

$obtainedSession = $repo->get();
new logger\Debug(json_encode($obtainedSession));
$repo->delete();

$obtainedSession = $repo->get();
new logger\Debug(json_encode($obtainedSession));
