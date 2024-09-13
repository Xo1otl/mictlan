<?php

require __DIR__ . '/middleware.php';

$session = authenticate();
getVoiceController()->deleteVoice($_POST, $session->accountId);
// echo "<a href='list_voice.php'>音声を削除しました。</a>";
header('Location: ./list_voice.php?status=complete_delete', true, 303);
