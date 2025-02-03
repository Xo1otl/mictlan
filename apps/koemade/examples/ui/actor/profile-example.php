<?php

use koemade\actor;
use koemade\dbadapter;
use koemade\dbadapter\exceptions;

require_once __DIR__ . '/../bootstrap.php';

$token = $_COOKIE['auth_token'] ?? '';
if (!$token) {
    echo "auth_tokenが設定されていません";
    exit;
}

try {
    $claims = $tokenService->verify($token);
    if (($claims->role ?? '') !== 'actor') {
        http_response_code(403);
        echo "actorしかアクセスできません";
        exit;
    }
} catch (Exception $e) {
    http_response_code(403);
    echo $e->getMessage();
    exit;
}

$repository = new dbadapter\QueryRepository();
try {
    $feed = $repository->actorFeed($claims->sub); // subは所有者を表すid
} catch (exceptions\ProfileNotFoundException $e) {
    $feed = null;
}

// プロファイル保存処理
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['save_profile'])) {
    // ファイルアップロード処理
    $uploadInfo = $storage->upload($_FILES['profile_image']);

    $profileInput = new actor\ProfileInput(
        $_POST['display_name'],
        $_POST['self_promotion'] ?? null,
        (int) $_POST['price'] ?? null,
        $_POST['status'] ?? null,
        isset($_POST['nsfw_allowed']) ? (bool) $_POST['nsfw_allowed'] : false,
        (int) $_POST['nsfw_price'] ?? null,
        isset($_POST['extreme_allowed']) ? (bool) $_POST['extreme_allowed'] : false,
        (int) $_POST['extreme_surcharge'] ?? null,
        $uploadInfo['type'],
        $uploadInfo['size'],
        $uploadInfo['path']
    );

    $profileService = new dbadapter\ProfileService();
    $profileService->save($_POST['user_id'], $profileInput);
    echo "プロファイルが保存されました！";
}

// パスワード変更処理
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['change_password'])) {
    $accountService = new dbadapter\AccountService();
    $accountService->changePassword($claims->sub, $_POST['current_password'], $_POST['new_password']);
    echo "パスワードが変更されました！";
}
?>

<!DOCTYPE html>
<html lang="ja">

<head>
    <meta charset="UTF-8">
    <title>編集機能デモ</title>
</head>

<body>
    <h1>プロファイル編集</h1>
    <form method="post" enctype="multipart/form-data" onsubmit="submitForm(event)">
        <input type="hidden" name="csrf_token" value="<?php echo $_SESSION['csrf_token'] ?>"> <!-- CSRFトークンを埋め込む -->
        <input type="hidden" name="user_id" value="<?= $claims->sub ?>">
        <label for="display_name">表示名:</label>
        <input type="text" id="display_name" name="display_name"
            value="<?= htmlspecialchars($feed->actor->name ?? '') ?>" required><br>
        <label for="self_promotion">自己PR:</label>
        <textarea id="self_promotion"
            name="self_promotion"><?= htmlspecialchars($feed->actor->description ?? '') ?></textarea><br>
        <label for="price">価格:</label>
        <input type="number" id="price" name="price"
            value="<?= htmlspecialchars($feed->actor->price->default ?? '') ?>"><br>
        <label for="status">ステータス:</label>
        <input type="text" id="status" name="status" value="<?= htmlspecialchars($feed->actor->status ?? '') ?>"><br>
        <label for="nsfw_allowed">NSFW許可:</label>
        <input type="checkbox" id="nsfw_allowed" name="nsfw_allowed" value="true" <?= isset($feed->actor->nsfwAllowed) && $feed->actor->nsfwAllowed ? 'checked' : '' ?>><br>
        <label for="nsfw_price">NSFW価格:</label>
        <input type="number" id="nsfw_price" name="nsfw_price"
            value="<?= htmlspecialchars($feed->actor->price->nsfw ?? '') ?>"><br>
        <label for="extreme_allowed">過激な内容許可:</label>
        <input type="checkbox" id="extreme_allowed" name="extreme_allowed" value="true"
            <?= isset($feed->actor->nsfwExtremeAllowed) && $feed->actor->nsfwExtremeAllowed ? 'checked' : '' ?>><br>
        <label for="extreme_surcharge">過激な内容追加料金:</label>
        <input type="number" id="extreme_surcharge" name="extreme_surcharge"
            value="<?= htmlspecialchars($feed->actor->price->nsfw_extreme ?? '') ?>"><br>
        <label for="profile_image">プロファイル画像:</label>
        <input type="file" id="profile_image" name="profile_image"><br>
        <?php if (!empty($feed->actor->avator_url)): ?>
            <img src="<?= htmlspecialchars($feed->actor->avator_url) ?>" alt="プロファイル画像" style="max-width: 200px;"><br>
        <?php endif; ?>
        <button type="submit" name="save_profile">プロファイルを保存</button>
    </form>

    <h1>パスワード変更</h1>
    <form method="post">
        <input type="hidden" name="csrf_token" value="<?php echo $_SESSION['csrf_token'] ?>"> <!-- CSRFトークンを埋め込む -->
        <label for="current_password">現在のパスワード:</label>
        <input type="password" id="current_password" name="current_password" required><br>
        <label for="new_password">新しいパスワード:</label>
        <input type="password" id="new_password" name="new_password" required><br>
        <button type="submit" name="change_password">パスワードを変更</button>
    </form>

    <script>
        function submitForm(event) {
            event.preventDefault(); // フォームのデフォルトの送信を防止

            const form = event.target;
            const formData = new FormData(form);

            // 送信ボタンの name を自動的に追加
            const submitButton = form.querySelector('button[type="submit"], input[type="submit"]');
            if (submitButton && submitButton.name) {
                formData.append(submitButton.name, submitButton.value || '1');
            }

            fetch(form.action, {
                method: form.method,
                body: formData,
            })
                .then(response => {
                    location.reload();
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        }
    </script>
</body>

</html>