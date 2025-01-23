<?php

require_once __DIR__ . "/../kernel/bootstrap.php";

use koemade\dbadapter;

$actorVoiceService = new dbadapter\ActorVoiceService();

$voice_id = "123";
$newTitle = "新しいタイトル";

// editTitle メソッドをテスト
try {
    $actorVoiceService->editTitle($voice_id, $newTitle);
    echo "editTitle メソッドのテストが成功しました。\n";
} catch (\Exception $e) {
    echo "editTitle メソッドのテストが失敗しました: " . $e->getMessage() . "\n";
}

$tagIds = [1, 2, 3];

// updateTags メソッドをテスト
try {
    $actorVoiceService->updateTags($voice_id, $tagIds);
    echo "updateTags メソッドのテストが成功しました。\n";
} catch (\Exception $e) {
    echo "updateTags メソッドのテストが失敗しました: " . $e->getMessage() . "\n";
}
