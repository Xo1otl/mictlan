<?php

require __DIR__ . '/../vendor/autoload.php';

function NewApp()
{
    $userRepo = new \auth\PdoAccountRepo();
    $sessionRepo = new \auth\FileSessionRepo();
    return new \auth\App($userRepo, $sessionRepo);
}

function Session(): ?\auth\Session
{
    $sessionRepo = new \auth\FileSessionRepo();
    return $sessionRepo->get();
}