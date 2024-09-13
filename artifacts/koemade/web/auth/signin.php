<?php
$message = '';
if( isset($_GET['err']) ) {
    $message = '<p class="notice error">ログインに失敗しました。</p>';
}elseif( isset($_GET['signout']) ) {
    require __DIR__ . '/middleware.php';
    $sessionRepo = new \filesystem\SessionRepo();
    $sessionRepo->delete();
    $message = '<p class="notice signout">ログアウトしました。</p>';
}
?>
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ログイン</title>

    <link rel="stylesheet" href="../assets/css/style.css">
    <link rel="stylesheet" href="../assets/css/fontawesome.min.css">
    <link rel="stylesheet" href="../assets/css/brands.min.css">
    <link rel="stylesheet" href="../assets/css/solid.min.css">
</head>
<body class="has-form">
<?= $message; ?>
<div class="container form signin">
    <h1>ログイン</h1>
    <form action="handle-signin.php" method="post" enctype="multipart/form-data">
        <label for="username">メールアドレス</label>
        <input type="email" id="username" name="username" placeholder="メールアドレスを入力" required>

        <label for="password">パスワード</label>
        <input type="password" id="password" name="password" placeholder="パスワードを入力" required>

        <button type="submit" class="primary">ログイン</button>
    </form>
</div>
</body>
</html>

