<?php

use koemade\dbadapter;
use koemade\util;
use koemade\auth;

require_once __DIR__ . "/../kernel/bootstrap.php";

$conn = dbadapter\DBConnection::getInstance();
$logger = util\logger::getInstance();

$secretKey = bin2hex(random_bytes(32));

$auth = new dbadapter\AuthService($conn, $secretKey);
$token = $auth->authenticate("admin@koemade.net", "Abcd1234*");
$logger->info("Token: $token");

$tokenService = new auth\JWTService($secretKey);
$claims = $tokenService->verify($token);
$logger->info("Claims: " . json_encode($claims));
