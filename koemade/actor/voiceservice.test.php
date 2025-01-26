<?php

require_once __DIR__ . "/../kernel/bootstrap.php";

use koemade\dbadapter;
use koemade\actor;

$actorVoiceService = new dbadapter\ActorVoiceService();

$sub = 3; // アカウントID

$newTitle = "テスト用の新しいボイス";
$newTagIds = [1, 2, 3]; // 新しいタグ

$input = new actor\NewVoiceInput($sub, $newTitle, 5, "audio/mpeg", path: "test.mp3");

try {
    $actorVoiceService->newVoice($input);
    echo "newVoice メソッドのテストが成功しました。\n";
} catch (\Exception $e) {
    echo "テストが失敗しました: " . $e->getMessage() . "\n";
}

$voice_id = 11;
$oldTitle = "古いタイトル";
$newTitle = "新しいタイトル";

// updateVoice メソッドをテスト（タイトルのみ更新）
try {
    $actorVoiceService->updateVoice(new actor\UpdateVoiceInput($sub, $voice_id, newTitle: $newTitle));
    echo "updateVoice メソッドのテスト（タイトルのみ更新）が成功しました。\n";
} catch (\Exception $e) {
    echo "updateVoice メソッドのテスト（タイトルのみ更新）が失敗しました: " . $e->getMessage() . "\n";
}

$originalTagIds = [1, 2, 3];
$newTagIds = [4, 5, 6];

// updateVoice メソッドをテスト（タグを置き換え）
try {
    $actorVoiceService->updateVoice(new actor\UpdateVoiceInput($sub, $voice_id, tagIds: $newTagIds));
    echo "updateVoice メソッドのテスト（タグを置き換え）が成功しました。\n";
} catch (\Exception $e) {
    echo "updateVoice メソッドのテスト（タグを置き換え）が失敗しました: " . $e->getMessage() . "\n";
}

// タグを元に戻すテスト
try {
    $actorVoiceService->updateVoice(new actor\UpdateVoiceInput($sub, $voice_id, tagIds: $originalTagIds));
    echo "タグを元に戻すテストが成功しました。\n";
} catch (\Exception $e) {
    echo "タグを元に戻すテストが失敗しました: " . $e->getMessage() . "\n";
}
