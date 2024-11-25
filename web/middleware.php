<?php

require __DIR__ . '/../vendor/autoload.php';

use koemade\auth;

function NewApp()
{
    $userRepo = new auth\PdoAccountRepo();
    $sessionRepo = new auth\FileSessionRepo();
    return new auth\Handler($userRepo, $sessionRepo);
}

function Session(): ?auth\Session
{
    $sessionRepo = new auth\FileSessionRepo();
    return $sessionRepo->get();
}