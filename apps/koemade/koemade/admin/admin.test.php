<?php

require_once __DIR__ . '/../kernel/bootstrap.php';

use koemade\dbadapter;
use koemade\admin;
use koemade\auth;

$accountService = new dbadapter\AccountService();
$input = new admin\CreateAccountInput("actor1@gmail.com", "password");
$accountService->createAccount($input);
$accountService->banAccount("actor1@gmail.com");

$auth = new dbadapter\AuthService($secretKey);
$token = $auth->loginAsAccount("actor1@gmail.com");
$logger->info("Token: $token");
$tokenService = new auth\JWTService($secretKey);
$claims = $tokenService->verify($token);
// role が actor で status が banned であることを確認
if ($claims->role === 'actor' && $claims->status === 'banned') {
    $logger->info("Claims: " . json_encode($claims));
} else {
    $logger->error("Invalid claims: " . json_encode($claims));
}

$accountService->deleteAccount("actor1@gmail.com");
