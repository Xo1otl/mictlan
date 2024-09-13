<?php

require __DIR__ . '/middleware.php';

$session = authenticate();

getProfileController()->handleUpdateProfile($_POST, $session->accountId);
