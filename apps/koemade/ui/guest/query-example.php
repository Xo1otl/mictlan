<?php

require_once __DIR__ . '/../bootstrap.php';

?>

<!DOCTYPE html>
<html lang="ja">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>声優と音声取得デモ</title>
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

        .result-item img {
            max-width: 100px;
            height: auto;
        }

        .result-item audio {
            width: 100%;
        }
    </style>
</head>

<body>
    <div class="container">
        <h1>声優と音声取得デモ</h1>

        <!-- 声優取得セクション -->
        <div class="search-section">
            <h2>声優取得</h2>
            <form id="actor-fetch-form">
                <label for="actor-id">声優ID:</label>
                <input type="text" id="actor-id" name="id" required>
                <button type="submit">取得</button>
            </form>
        </div>

        <!-- 音声取得セクション -->
        <div class="search-section">
            <h2>音声取得</h2>
            <form id="voice-fetch-form">
                <label for="voice-id">音声ID:</label>
                <input type="text" id="voice-id" name="id" required>
                <button type="submit">取得</button>
            </form>
        </div>

        <!-- 結果表示セクション -->
        <div class="results">
            <h2>取得結果</h2>
            <div id="actor-result"></div>
            <div id="voice-result"></div>
        </div>
    </div>

    <script>
        // 声優取得フォームの処理
        document.getElementById('actor-fetch-form').addEventListener('submit', function (event) {
            event.preventDefault();
            const actorId = document.getElementById('actor-id').value;
            fetch(`${apiURL}/actor/${actorId}`)
                .then(response => response.json())
                .then(data => {
                    const resultDiv = document.getElementById('actor-result');
                    resultDiv.innerHTML = '<h3>声優情報</h3>';
                    const item = document.createElement('div');
                    item.className = 'result-item';
                    item.innerHTML = `
                        <h3>${data.actor.name}</h3>
                        <p>ID: ${data.actor.id}</p>
                        <p>ステータス: ${data.actor.status}</p>
                        <p>ランク: ${data.actor.rank}</p>
                        <p>説明: ${data.actor.description}</p>
                        <img src="${data.actor.avator_url}" alt="${data.actor.name}">
                        <h4>サンプル音声</h4>
                        <p>総数: ${data.sample_voices.total_matches}</p>
                        <ul>
                            ${data.sample_voices.items.map(voice => `
                                <li>
                                    <p>ID: ${voice.id}</p>
                                    <p>名前: ${voice.name}</p>
                                    <audio controls>
                                        <source src="${voice.source_url}" type="audio/mpeg">
                                        お使いのブラウザは音声をサポートしていません。
                                    </audio>
                                </li>
                            `).join('')}
                        </ul>
                    `;
                    resultDiv.appendChild(item);
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        });

        // 音声取得フォームの処理
        document.getElementById('voice-fetch-form').addEventListener('submit', function (event) {
            event.preventDefault();
            const voiceId = document.getElementById('voice-id').value;
            fetch(`${apiURL}/voice/${voiceId}`)
                .then(response => response.json())
                .then(data => {
                    const resultDiv = document.getElementById('voice-result');
                    resultDiv.innerHTML = '<h3>音声情報</h3>';
                    const item = document.createElement('div');
                    item.className = 'result-item';
                    item.innerHTML = `
                    <h3>${data.title}</h3>
                    <p>ID: ${data.id}</p>
                    <p>アカウント: ${data.account.username}</p>
                    <p>ファイル名: ${data.filename}</p>
                    <p>作成日時: ${data.created_at}</p>
                    <h4>タグ</h4>
                    <ul>
                        ${data.tags.map(tag => `
                            <li>
                                <p>カテゴリ: ${tag.category}</p>
                                <p>名前: ${tag.name}</p>
                            </li>
                        `).join('')}
                    </ul>
                    <audio controls>
                        <source src="${data.filename}" type="audio/mpeg">
                        お使いのブラウザは音声をサポートしていません。
                    </audio>
                `;
                    resultDiv.appendChild(item);
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        });
    </script>
</body>

</html>