<?php

use koemade\dbadapter;
use koemade\auth;

require_once __DIR__ . "/../kernel/bootstrap.php";

$authService = new dbadapter\AuthService($secretKey);
$token = $authService->authenticate("admin@koemade.net", "Abcd1234*");
$logger->info("Token: $token");

$tokenService = new auth\JWTService($secretKey);
$claims = $tokenService->verify($token);
$logger->info("Claims: " . json_encode($claims));
