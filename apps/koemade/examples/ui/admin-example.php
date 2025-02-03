<?php

require_once __DIR__ . '/bootstrap.php';

use koemade\dbadapter;
use koemade\admin;
use koemade\auth;

// 認証チェック
$token = $_COOKIE['auth_token'] ?? '';
if (!$token) {
    echo "auth_tokenが設定されていません";
    exit;
}

try {
    $tokenService = new auth\JWTService($secretKey);
    $claims = $tokenService->verify($token);
    if (($claims->role ?? '') !== 'admin') {
        http_response_code(403);
        echo "adminしかアクセスできません";
        exit;
    }
} catch (Exception $e) {
    http_response_code(403);
    echo $e->getMessage();
    exit;
}

// アカウント管理機能
$accountService = new dbadapter\AccountService();
$auth = new dbadapter\AuthService($secretKey);

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $email = $_POST['email'];
    $password = $_POST['password'];

    try {
        if (isset($_POST['create'])) {
            if (empty($password)) {
                $logger->error("Password is required for account creation.");
            } else {
                $input = new admin\CreateAccountInput($email, $password);
                $accountService->createAccount($input);
                $logger->info("Account created: $email");
            }
        } elseif (isset($_POST['ban'])) {
            $accountService->banAccount($email);
            $logger->info("Account banned: $email");
        } elseif (isset($_POST['login'])) {
            // ログイン処理
            $token = $auth->loginAsAccount($email); // パスワードを引数に追加

            // トークンをHttpOnlyクッキーに保存
            setcookie('auth_token', $token, [
                'expires' => time() + 3600, // 1時間有効
                'path' => '/',
                // 'secure' => true, // HTTPSのみ (対応後に有効化)
                'httponly' => true, // JavaScriptからアクセス不可
                'samesite' => 'Strict' // CSRF対策
            ]);

            // ログイン成功後にリダイレクト
            header('Location: actor/profile-example.php');
            exit;
        } elseif (isset($_POST['delete'])) {
            $accountService->deleteAccount($email);
            $logger->info("Account deleted: $email");
        }
    } catch (Exception $e) {
        echo $e->getMessage();
    }
}

?>

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <title>Account Management</title>
</head>

<body>
    <h1>Account Management</h1>
    <p>Hello, admin!</p>
    <form method="POST">
        <input type="hidden" name="csrf_token" value="<?php echo $_SESSION['csrf_token'] ?>"> <!-- CSRFトークンを埋め込む -->
        <label for="email">Email:</label>
        <input type="email" id="email" name="email" required>
        <br>
        <label for="password">Password:</label>
        <input type="password" id="password" name="password">
        (アカウント作成時のみ入力が必要)
        <br>
        <button type="submit" name="create">Create Account</button>
        <button type="submit" name="ban">Ban Account</button>
        <button type="submit" name="login">Login</button>
        <button type="submit" name="delete">Delete Account</button>
    </form>
</body>

</html>