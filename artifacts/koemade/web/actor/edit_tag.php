<?php

require __DIR__ . '/middleware.php';

$session = authenticate();
$controller = getVoiceController();

$controller->editTag($_POST, $session->accountId);
// echo "edit tag";
header('Location: ./list_voice.php?status=complete_edit_tag', true, 303);

