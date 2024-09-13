<?php
require __DIR__ . '/middleware.php';

$session = getSession();
$message = '';

if( isset($_GET['err']) ) {
    $message = '<p class="notice error">記入内容に不備があります。</p>';
}

if ($session->role != \auth\Role::ADMIN) {
    http_response_code(400);
    exit('Access Denied');
    // echo "Access Denied";
    // exit();
}
?>

<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>アカウント作成</title>
    <link rel="stylesheet" href="../assets/css/style.css">
    <link rel="stylesheet" href="../assets/css/fontawesome.min.css">
    <link rel="stylesheet" href="../assets/css/brands.min.css">
    <link rel="stylesheet" href="../assets/css/solid.min.css">
</head>
<body class="mypage">
<?= $message; ?>
<?php require('../inc/header.php'); ?>
<div class="container form signup">
    <h1>アカウント作成</h1>
    <form action="handle-signup.php" method="post" enctype="multipart/form-data">
        <label for="email">メールアドレス</label>
        <input type="email" id="email" name="email" placeholder="メールアドレスを入力" required>

        <label for="password">パスワード</label>
        <input type="password" id="password" name="password" placeholder="パスワードを入力" required>
        <p class="note"><small>パスワードは最低8文字以上、半角英字の大文字・小文字・数字と一部記号（!@#$%^&*()）をそれぞれ1文字以上含む必要があります。</small></p>

        <button type="submit" class="primary">作成</button>
    </form>
</div>
</body>
</html>
