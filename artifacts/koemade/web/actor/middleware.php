<?php

require __DIR__ . '/../../vendor/autoload.php';

function authenticate(): \auth\Session
{
    $sessionRepo = new \filesystem\SessionRepo();
    $session = $sessionRepo->get();

    if ($session->role->value !== \auth\Role::ACTOR && $session->role->value !== \auth\Role::ADMIN) {
        http_response_code(400);
        exit('Access Denied');
        // echo "Access Denied";
        // exit();
    }

    return $session;
}

function getProfileController(): \actor\Controller
{
    $profileRepo = new \mysql\ProfileRepo();
    $app = new \actor\App($profileRepo);
    return new \actor\Controller($app);
}

function getVoiceController(): \voice\Controller
{
    $repo = new \mysql\VoiceRepo();
    $app = new \voice\App($repo);
    return new \voice\Controller($app);
}
