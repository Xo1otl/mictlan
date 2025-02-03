<?php
use koemade\dbadapter;
use koemade\dbadapter\exceptions;
use koemade\actor;

require_once __DIR__ . '/../bootstrap.php';

$token = $_COOKIE['auth_token'] ?? '';
if (!$token)
    die("auth_tokenが設定されていません");

try {
    $claims = $tokenService->verify($token);
    if (($claims->role ?? '') !== 'actor')
        die("actorしかアクセスできません");
} catch (Exception $e) {
    die($e->getMessage());
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (isset($_POST['voice_id'])) {
        // 既存の音声ファイルの更新処理
        $voice_id = $_POST['voice_id'] ?? null;
        $newTagIds = array_map('intval', $_POST['tags'] ?? []);
        $newTitle = $_POST['new_title'] ?? null;

        if ($voice_id) {
            $actorVoiceService = new dbadapter\ActorVoiceService();
            try {
                $actorVoiceService->updateVoice(new actor\UpdateVoiceInput($claims->sub, $voice_id, tagIds: $newTagIds, newTitle: $newTitle));
                header("Location: " . $_SERVER['PHP_SELF']);
                exit;
            } catch (\Exception $e) {
                die("更新に失敗しました: " . $e->getMessage());
            }
        }
    } else {
        // 新しい音声ファイルのアップロード処理
        $newTitle = $_POST['new_title'] ?? null;
        $newTagIds = array_map('intval', $_POST['tags'] ?? []);
        $file = $_FILES['voice_file'] ?? null;

        if ($newTitle && $file && $file['error'] === UPLOAD_ERR_OK) {
            try {
                $uploadInfo = $storage->upload($file);
                $mime_type = $uploadInfo['type'];
                $uploadPath = $uploadInfo['path'];

                $input = new actor\NewVoiceInput($claims->sub, $newTitle, $uploadInfo['size'], $mime_type, path: $uploadPath, tagIds: $newTagIds);
                $actorVoiceService = new dbadapter\ActorVoiceService();
                $actorVoiceService->newVoice($input);

                header("Location: " . $_SERVER['PHP_SELF']);
                exit;
            } catch (\Exception $e) {
                die("アップロードに失敗しました: " . $e->getMessage());
            }
        } else {
            die("ファイルのアップロードに失敗しました。");
        }
    }
}

$repository = new dbadapter\QueryRepository();
$tags = $repository->listAllTags();
try {
    $feed = $repository->actorFeed($claims->sub);
} catch (exceptions\ProfileNotFoundException $e) {
    $feed = null;
}
?>

<!DOCTYPE html>
<html>

<head>
    <style>
        .tag {
            display: inline-block;
            margin: 2px;
            padding: 2px 5px;
            background: #eee;
            cursor: pointer;
        }

        .selected {
            background: #ccc;
        }
    </style>
</head>

<body>
    <!-- 新しい音声ファイルのアップロードフォーム -->
    <h3>新しいボイスを追加</h3>
    <hr width="100%" size="2">
    <form method="post" enctype="multipart/form-data">
        <input type="hidden" name="csrf_token" value="<?php echo $_SESSION['csrf_token'] ?>"> <!-- CSRFトークンを埋め込む -->
        <div>
            <label for="new_title">タイトル:</label>
            <input type="text" name="new_title" id="new_title" required>
        </div>
        <div>
            <label for="voice_file">ファイル:</label>
            <input type="file" name="voice_file" id="voice_file" required>
        </div>
        <div>
            <?php foreach ($tags as $tag): ?>
                <span class="tag" data-id="<?= $tag->id ?>"
                    onclick="toggleTag(this)"><?= htmlspecialchars($tag->name) ?></span>
            <?php endforeach ?>
        </div>
        <div id="tags-container_new">
            <!-- 選択されたタグがここに追加されます -->
        </div>
        <input type="submit" value="アップロード">
    </form>

    <h3>既存のボイス一覧&編集</h3>
    <hr width="100%" size="2">
    <!-- 既存の音声ファイルの編集フォーム -->
    <?php if ($feed && !empty($feed->sampleVoices)): ?>
        <?php foreach ($feed->sampleVoices as $voice): ?>
            <form method="post">
                <input type="hidden" name="csrf_token" value="<?php echo $_SESSION['csrf_token'] ?>"> <!-- CSRFトークンを埋め込む -->
                <input type="hidden" name="voice_id" value="<?= $voice->id ?>">
                <h3><input name="new_title" value="<?= htmlspecialchars($voice->name) ?>"></h3>
                <div>
                    <?php foreach ($tags as $tag): ?>
                        <span class="tag <?= in_array($tag->id, array_column($voice->tags, 'id')) ? 'selected' : '' ?>"
                            data-id="<?= $tag->id ?>" onclick="toggleTag(this)"><?= htmlspecialchars($tag->name) ?></span>
                    <?php endforeach ?>
                </div>
                <div id="tags-container_<?= $voice->id ?>">
                    <?php foreach ($voice->tags as $tag): ?>
                        <input type="hidden" name="tags[]" value="<?= $tag->id ?>">
                    <?php endforeach ?>
                </div>
                <input type="submit" value="Save">
            </form>
            <a href="<?= htmlspecialchars($voice->source_url) ?>">Source URL</a>
        <?php endforeach ?>
    <?php else: ?>
        <p>No voices available.</p>
    <?php endif; ?>

    <script>
        function toggleTag(el) {
            el.classList.toggle('selected');
            const form = el.closest('form');
            const tagsContainer = form.querySelector('#tags-container_new') || form.querySelector('#tags-container_' + form.querySelector('input[name="voice_id"]').value);
            const tagId = el.dataset.id;

            const existingInput = tagsContainer.querySelector(`input[value="${tagId}"]`);

            if (existingInput) {
                tagsContainer.removeChild(existingInput);
            } else {
                const newInput = document.createElement('input');
                newInput.type = 'hidden';
                newInput.name = 'tags[]';
                newInput.value = tagId;
                tagsContainer.appendChild(newInput);
            }
        }
    </script>
</body>

</html>