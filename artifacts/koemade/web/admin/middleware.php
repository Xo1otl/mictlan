<?php

require __DIR__ . '/../../vendor/autoload.php';

function authenticateAdmin(): \auth\Session
{
    $sessionRepo = new \filesystem\SessionRepo();
    $session = $sessionRepo->get();

    if ($session->role->value !== \auth\Role::ADMIN) {
        http_response_code(400);
        exit('Access Denied');
    }

    return $session;
}

function getProfileController(): \actor\Controller
{
    $profileRepo = new \mysql\ProfileRepo();
    $app = new \actor\App($profileRepo);
    return new \actor\Controller($app);
}

function getAuthController(): \auth\Controller
{
    $keyService = new \bcrypt\KeyService();
    $userRepo = new \mysql\AccountRepo();
    $sessionRepo = new \filesystem\SessionRepo();
    $authApp = new \auth\App($keyService, $userRepo, $sessionRepo);
    return new \auth\Controller($authApp);
}
function getVoiceController(): \voice\Controller
{
    $repo = new \mysql\VoiceRepo();
    $app = new \voice\App($repo);
    return new \voice\Controller($app);
}
