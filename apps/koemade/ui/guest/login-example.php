<?php

require_once __DIR__ . '/../bootstrap.php';

// ログイン処理
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $email = $_POST['email'] ?? '';
    $password = $_POST['password'] ?? '';

    try {
        // 認証サービスを使ってログイン
        $token = $authService->authenticate($email, $password);

        // トークンをHttpOnlyクッキーに保存
        setcookie('auth_token', $token, [
            'expires' => time() + 3600, // 1時間有効
            'path' => '/',
            'secure' => true, // HTTPSのみ
            'httponly' => true, // JavaScriptからアクセス不可
            'samesite' => 'Strict' // CSRF対策
        ]);

        // ログイン成功後にリダイレクト
        header('Location: ' . $_SERVER['PHP_SELF']);
        exit;
    } catch (Exception $e) {
        $error = "Login failed: " . $e->getMessage();
    }
}

// ログアウト処理
if (isset($_GET['logout'])) {
    // クッキーを削除 (HttpOnly属性を付与しているため、JavaScriptから削除できない)
    setcookie('auth_token', '', [
        'expires' => time() - 3600, // 過去の日付に設定して削除
        'path' => '/',
        'secure' => true,
        'httponly' => true,
        'samesite' => 'Strict'
    ]);

    // ログアウト後にリダイレクト
    header('Location: ' . $_SERVER['PHP_SELF']);
    exit;
}

// ログイン状態の確認
$isLoggedIn = false;
$claims = null;

// クッキーからトークンを取得
$token = $_COOKIE['auth_token'] ?? '';
if (!empty($token)) {
    try {
        // トークン検証サービスを使ってトークンを検証
        $claims = $tokenService->verify($token);
        $isLoggedIn = true;
    } catch (Exception $e) {
        // 無効なトークンの場合は何もしない
    }
}

// ページ表示
if ($isLoggedIn) {
    echo "Welcome, " . htmlspecialchars($claims->role) . "!<br>";
    echo '<a href="?logout=1">Logout</a>';
} else {
    // ログインフォーム
    if (isset($error)) {
        echo "<p style='color: red;'>$error</p>";
    }
    echo <<<HTML
    <form method="post">
        <input type="hidden" name="csrf_token" value="{$_SESSION['csrf_token']}"> <!-- CSRFトークンを埋め込む -->
        Email: <input type="email" name="email" required><br>
        Password: <input type="password" name="password" required><br>
        <input type="submit" value="Login">
    </form>
    HTML;
}
