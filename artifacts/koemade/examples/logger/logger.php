<?php

require __DIR__ . '/../../vendor/autoload.php';

$logger = new \Monolog\Logger('my_logger');

$logger->pushHandler(new \Monolog\Handler\StreamHandler('php://stdout', \Monolog\Logger::DEBUG));

$test = ["key" => "value", "number" => 42];
$logger->debug('Debugging info', ['variable' => $test]);
