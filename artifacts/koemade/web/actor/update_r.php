<?php

require __DIR__ . '/middleware.php';

$session = authenticate();

getProfileController()->handleUpdateR($_POST, $session->accountId);
