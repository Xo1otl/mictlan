<?php

require_once __DIR__ . '/../../vendor/autoload.php';
require_once __DIR__ . '/secrets.php';

use koemade\util;
use koemade\dbadapter;
use koemade\auth;

$logger = util\Logger::getInstance();
$logger->info('Bootstraping the application');
$authService = new dbadapter\AuthService($secretKey);
$tokenService = new auth\JWTService($secretKey);
