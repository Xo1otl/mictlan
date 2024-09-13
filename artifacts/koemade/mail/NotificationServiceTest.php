<?php

namespace mail;

require __DIR__ . '/../vendor/autoload.php';

$r = file_get_contents(__DIR__ . '/signupRequestExample.txt');
$r = unserialize($r);
$s = new NotificationService();
$s->notify($r);
