<!-- TODO: 声優が使用する機能をすべて一つのページにまとめたものを作成する、ログイン、プロフィール編集(jsonで直でいじればフォームいらん)、音声の投稿、音声の編集、タグ付けなど(これらもexampleではjsonで直でいじればおｋ) -->
<?php
// 既存の authService と tokenService を使う
require_once __DIR__ . "/../koemade/kernel/bootstrap.php";

// ログアウト処理
if (isset($_GET['logout'])) {
    setcookie('auth_token', '', time() - 3600, '/'); // クッキーを削除
    header('Location: ' . $_SERVER['PHP_SELF']); // リダイレクトしてリロード
    exit;
}

// ログイン処理
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $email = $_POST['email'] ?? '';
    $password = $_POST['password'] ?? '';

    try {
        // 認証サービスを使ってログイン
        $token = $authService->authenticate($email, $password);

        // トークンをクッキーに保存
        setcookie('auth_token', $token, time() + 3600, '/'); // 1時間有効
        header('Location: ' . $_SERVER['PHP_SELF']); // リダイレクトしてリロード
        exit;
    } catch (Exception $e) {
        $error = "Login failed: " . $e->getMessage();
    }
}

// ログイン状態の確認
$isLoggedIn = false;
$claims = null;

if (isset($_COOKIE['auth_token'])) {
    try {
        // トークン検証サービスを使ってトークンを検証
        $claims = $tokenService->verify($_COOKIE['auth_token']);
        $isLoggedIn = true;
    } catch (Exception $e) {
        setcookie('auth_token', '', time() - 3600, '/'); // 無効なトークンを削除
    }
}

// ページ表示
if ($isLoggedIn) {
    // ログイン中の表示
    echo "Welcome, " . htmlspecialchars($claims->role) . "!<br>";
    echo '<a href="?logout=1">Logout</a>'; // ログアウトボタン
} else {
    // ログインフォーム
    if (isset($error)) {
        echo "<p style='color: red;'>$error</p>";
    }
    echo <<<HTML
    <form method="post">
        Email: <input type="email" name="email" required><br>
        Password: <input type="password" name="password" required><br>
        <input type="submit" value="Login">
    </form>
    HTML;
}
