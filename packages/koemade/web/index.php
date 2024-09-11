<?php
require __DIR__ . '/middleware.php';

$session = Session();

function getContent($role, $username = '')
{
    $escapedUsername = htmlspecialchars($username, ENT_QUOTES, 'UTF-8');

    switch ($role) {
        case 'admin':
            return <<<HTML
            <h2>管理者ダッシュボード</h2>
            <p>ようこそ、管理者様。システムの管理機能にアクセスできます。</p>
            <ul>
                <li><a href='./signup.php'>新規ユーザー登録</a></li>
            </ul>
            HTML;

        case 'actor':
            return <<<HTML
            <h2>俳優ポータル</h2>
            <p>ようこそ、{$escapedUsername}さん。あなたの俳優ポータルへアクセスしました。</p>
            <ul>
                <li><a href='#'>プロフィール</a></li>
                <li><a href='#' id="signOut">サインアウト</a></li>
            </ul>
            <script>
            document.addEventListener('DOMContentLoaded', function() {
                const signoutLink = document.getElementById('signOut');
                if (signoutLink) {
                    signoutLink.addEventListener('click', function(e) {
                        e.preventDefault();

                        // すべてのクッキーを削除
                        document.cookie.split(";").forEach(function(c) { 
                            document.cookie = c.replace(/^ +/, "").replace(/=.*/, "=;expires=" + new Date().toUTCString() + ";path=/"); 
                        });

                        // ローカルストレージとセッションストレージをクリア
                        localStorage.clear();
                        sessionStorage.clear();

                        // ページをリロード
                        window.location.reload();
                    });
                }
            });
            </script>
            HTML;

        case 'guest':
        default:
            return <<<HTML
            <h2>ゲストページ</h2>
            <p>現在、ゲストとしてアクセスしています。より多くの機能を利用するには、ログインまたは登録が必要です。</p>
            <ul>
                <li><a href='./signin.php'>ログイン</a></li>
                <li><a href='#'>サービスについて</a></li>
            </ul>
            HTML;
    }
}

$content = getContent($session->role->value, $session->username ?? '');
?>

<!DOCTYPE html>
<html lang="ja">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title><?php echo ucfirst($session->role->value); ?> ページ</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            padding: 20px;
        }

        h2 {
            color: #333;
        }

        ul {
            list-style-type: none;
            padding: 0;
        }

        li {
            margin-bottom: 10px;
        }

        a {
            color: #0066cc;
            text-decoration: none;
        }

        a:hover {
            text-decoration: underline;
        }
    </style>
</head>

<body>
    <?php echo $content; ?>
</body>

</html>