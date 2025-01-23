<?php

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
$tags = $repository->listAllTags();
try {
    $feed = $repository->actorFeed($claims->sub); // subは所有者を表すid
} catch (exceptions\ProfileNotFoundException $e) {
    $feed = null;
}
?>

<!DOCTYPE html>
<html lang="ja">

<head>
    <meta charset="UTF-8">
    <title>Voice Editor</title>
    <style>
        .tag {
            display: inline-block;
            margin: 2px;
            padding: 2px 5px;
            background: #e0e0e0;
            border-radius: 3px;
        }

        .tag .remove {
            cursor: pointer;
            margin-left: 5px;
        }

        .all-tags .tag {
            cursor: pointer;
        }

        .all-tags .tag.disabled {
            background: #aaa;
            cursor: not-allowed;
        }
    </style>
</head>

<body>
    <h1>Voice Editor</h1>
    <?php if ($feed && !empty($feed->sampleVoices)): ?>
        <?php foreach ($feed->sampleVoices as $voice): ?>
            <div class="voice-item" data-voice-id="<?php echo $voice->id; ?>">
                <h3>Edit Voice: <?php echo htmlspecialchars($voice->name, ENT_QUOTES, 'UTF-8'); ?></h3>
                <form method="post" class="voice-form">
                    <input type="hidden" name="csrf_token" value="<?php echo $_SESSION['csrf_token'] ?>">
                    <input type="hidden" name="id" value="<?php echo htmlspecialchars($voice->id, ENT_QUOTES, 'UTF-8'); ?>">

                    <!-- 現在のタグ表示エリア -->
                    <div class="tag-list">
                        <?php foreach ($voice->tags as $tag): ?>
                            <div class="tag">
                                <?php echo htmlspecialchars($tag->name, ENT_QUOTES, 'UTF-8'); ?>
                                <span class="remove" data-tag-id="<?php echo $tag->id; ?>">×</span>
                            </div>
                        <?php endforeach; ?>
                    </div>

                    <!-- タグ追加エリア -->
                    <div class="all-tags">
                        <?php foreach ($tags as $tag): ?>
                            <?php $isSelected = in_array($tag, $voice->tags); ?>
                            <div class="tag <?php echo $isSelected ? 'disabled' : ''; ?>" data-tag-id="<?php echo $tag->id; ?>">
                                <?php echo htmlspecialchars($tag->name, ENT_QUOTES, 'UTF-8'); ?>
                            </div>
                        <?php endforeach; ?>
                    </div>

                    <button type="submit">Save Changes</button>
                </form>
            </div>
        <?php endforeach; ?>
    <?php else: ?>
        <p>No voices available.</p>
    <?php endif; ?>

    <script>
        // タグを取り外す処理
        document.querySelectorAll('.tag-list .remove').forEach(removeButton => {
            removeButton.addEventListener('click', () => {
                const tagId = removeButton.getAttribute('data-tag-id');
                const tag = removeButton.closest('.tag');
                const allTags = tag.closest('.voice-item').querySelector('.all-tags');

                // タグをUIから削除
                tag.remove();

                // タグ追加エリアで該当タグを再度選択可能にする
                const tagToEnable = allTags.querySelector(`.tag[data-tag-id="${tagId}"]`);
                if (tagToEnable) tagToEnable.classList.remove('disabled');
            });
        });

        // タグを追加する処理
        document.querySelectorAll('.all-tags .tag').forEach(tag => {
            tag.addEventListener('click', () => {
                if (tag.classList.contains('disabled')) return;

                const tagId = tag.getAttribute('data-tag-id');
                const tagName = tag.textContent;
                const tagList = tag.closest('.voice-item').querySelector('.tag-list');

                // タグをUIに追加
                const newTag = document.createElement('div');
                newTag.className = 'tag';
                newTag.innerHTML = `${tagName} <span class="remove" data-tag-id="${tagId}">×</span>`;
                tagList.appendChild(newTag);

                // タグ追加エリアで該当タグを選択不可にする
                tag.classList.add('disabled');

                // 新しいタグにイベントリスナーを追加
                newTag.querySelector('.remove').addEventListener('click', () => {
                    newTag.remove();
                    tag.classList.remove('disabled');
                });
            });
        });

        // フォーム送信時の処理
        document.querySelectorAll('.voice-form').forEach(form => {
            form.addEventListener('submit', (e) => {
                e.preventDefault();

                const voiceId = form.querySelector('input[name="id"]').value;
                const tags = Array.from(form.querySelectorAll('.tag-list .tag')).map(tag => {
                    return tag.querySelector('.remove').getAttribute('data-tag-id');
                });

                // サーバーに送信するデータ
                const data = {
                    voice_id: voiceId,
                    tags: tags,
                    csrf_token: form.querySelector('input[name="csrf_token"]').value
                };

                // サーバーにリクエストを送信
                fetch('/update-tags', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data),
                }).then(response => {
                    if (response.ok) alert('タグが正常に更新されました。');
                    else alert('タグの更新に失敗しました。');
                });
            });
        });
    </script>
</body>

</html>