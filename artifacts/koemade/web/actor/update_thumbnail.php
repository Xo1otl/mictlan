<?php

require __DIR__ . '/middleware.php';

$session = authenticate();

$storage = new \filesystem\ActorStorage();
try {
    getProfileController()->handleUpdateThumbnail($storage, $_FILES, $session->accountId);
} catch (\Exception $e) {
    echo $e->getMessage();
}
