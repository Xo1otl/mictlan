<?php

require __DIR__ . '/middleware.php';

authenticateAdmin();

$controller = getVoiceController();

$actorId = new \common\Id($_POST['actor_id']);
$controller->editTag($_POST, $actorId); 
// echo "edit tag";
header("Location: ./list_voice.php?status=complete_edit_tag&actor_id=$actorId", true, 303);
