<?php

require_once __DIR__ . '/../../vendor/autoload.php';
require_once __DIR__ . '/../auth/secrets.php';

use koemade\util;

$logger = util\Logger::getInstance();
$logger->info('Bootstraping the application');
