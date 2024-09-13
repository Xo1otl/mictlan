<?php

require __DIR__ . '/../../vendor/autoload.php';

function getController(): \auth\Controller
{
    $keyService = new \bcrypt\KeyService();
    $userRepo = new \mysql\AccountRepo();
    $sessionRepo = new \filesystem\SessionRepo();
    $authApp = new \auth\App($keyService, $userRepo, $sessionRepo);
    return new \auth\Controller($authApp);
}

function getSession(): ?\auth\Session
{
    $sessionRepo = new \filesystem\SessionRepo();
    return $sessionRepo->get();
}
