<?php

require __DIR__ . '/middleware.php';

$controller = getController();
$controller->handleSignin($_POST);

$session = getSession();
$sessionRepo = new \filesystem\SessionRepo();
$sessionRepo->delete();