<?php

require __DIR__ . '/middleware.php';

$session = authenticate();

$storage = new \filesystem\VoiceStorage();
var_dump($_POST);
var_dump($_FILES);
getVoiceController()->upload($storage, $_POST, $_FILES, $session->accountId);
