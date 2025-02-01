<?php

require_once __DIR__ . '/../bootstrap.php';
?>

<!DOCTYPE html>
<html lang="ja">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>声優と音声検索デモ</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }

        .container {
            max-width: 800px;
            margin: auto;
        }

        .search-section {
            margin-bottom: 20px;
        }

        .results {
            margin-top: 20px;
        }

        .result-item {
            border: 1px solid #ccc;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
        }

        .result-item h3 {
            margin-top: 0;
        }

        .result-item p {
            margin: 5px 0;
        }
    </style>
</head>

<body>
    <div class="container">
        <h1>声優と音声検索デモ</h1>

        <!-- 声優検索セクション -->
        <div class="search-section">
            <h2>声優検索</h2>
            <form id="actor-search-form">
                <label for="actor-name">名前:</label>
                <input type="text" id="actor-name" name="name_like">
                <label for="actor-status">ステータス:</label>
                <input type="text" id="actor-status" name="status" value="受付中">
                <label for="nsfw-allowed">NSFW許可:</label>
                <select id="nsfw-allowed" name="nsfw_allowed">
                    <option value="0">許可しない</option>
                    <option value="1">許可する</option>
                </select>
                <button type="submit">検索</button>
            </form>
        </div>

        <!-- 音声検索セクション -->
        <div class="search-section">
            <h2>音声検索</h2>
            <form id="voice-search-form">
                <label for="voice-title">タイトル:</label>
                <input type="text" id="voice-title" name="title">
                <label for="voice-tags">タグ:</label>
                <input type="text" id="voice-tags" name="tags[]" placeholder="genre:vote,theme:staff">
                <button type="submit">検索</button>
            </form>
        </div>

        <!-- 検索結果表示セクション -->
        <div class="results">
            <h2>検索結果</h2>
            <div id="actor-results"></div>
            <div id="voice-results"></div>
        </div>
    </div>

    <script>
        // 声優検索フォームの処理
        document.getElementById('actor-search-form').addEventListener('submit', function (event) {
            event.preventDefault();
            const formData = new FormData(event.target);
            const params = new URLSearchParams(formData).toString();
            fetch(`//api.koemade.net/search-actors?${params}`)
                .then(response => response.json())
                .then(data => {
                    const resultsDiv = document.getElementById('actor-results');
                    resultsDiv.innerHTML = '<h3>声優検索結果</h3>';
                    data.items.forEach(actor => {
                        const item = document.createElement('div');
                        item.className = 'result-item';
                        item.innerHTML = `
                            <h3>${actor.name}</h3>
                            <p>ID: ${actor.id}</p>
                            <p>ステータス: ${actor.status}</p>
                            <p>ランク: ${actor.rank}</p>
                            <img src="${actor.avatar_url}" alt="${actor.name}" width="100">
                        `;
                        resultsDiv.appendChild(item);
                    });
                });
        });

        // 音声検索フォームの処理
        document.getElementById('voice-search-form').addEventListener('submit', function (event) {
            event.preventDefault();
            const formData = new FormData(event.target);

            // tagsが空の場合はtagsキーを削除
            if (!formData.get('tags[]')) {
                formData.delete('tags[]');
            }

            const params = new URLSearchParams(formData).toString();
            fetch(`//api.koemade.net/search-voices?${params}`)
                .then(response => response.json())
                .then(data => {
                    const resultsDiv = document.getElementById('voice-results');
                    resultsDiv.innerHTML = '<h3>音声検索結果</h3>';
                    data.items.forEach(voice => {
                        const item = document.createElement('div');
                        item.className = 'result-item';
                        item.innerHTML = `
                            <h3>${voice.name}</h3>
                            <p>ID: ${voice.id}</p>
                            <p>声優: ${voice.actor.name} (ID: ${voice.actor.id})</p>
                            <p>タグ: ${voice.tags.map(tag => `${tag.category}:${tag.name}`).join(', ')}</p>
                            <audio controls>
                                <source src="${voice.source_url}" type="audio/mpeg">
                                お使いのブラウザは音声をサポートしていません。
                            </audio>
                        `;
                        resultsDiv.appendChild(item);
                    });
                });
        });
    </script>
</body>

</html>