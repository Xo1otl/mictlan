<?php

require __DIR__ . '/middleware.php';

$session = getSession();

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
    <title>アカウントBAN</title>
    <link rel="stylesheet" href="../assets/css/style.css">
    <link rel="stylesheet" href="../assets/css/fontawesome.min.css">
    <link rel="stylesheet" href="../assets/css/brands.min.css">
    <link rel="stylesheet" href="../assets/css/solid.min.css">
</head>
<body class="mypage">
<?php require('../inc/header.php'); ?>
<div class="container form delete-account">
    <h1>アカウントBan</h1>
    <form action="handle-ban.php" method="post" enctype="multipart/form-data">
        <label for="username">ユーザーネーム(メールアドレス)</label>
        <input type="email" id="username" name="username" placeholder="ユーザーネームを入力" required>
        <button type="submit" class="primary">Banする</button>
    </form>
</div>
</body>
</html>
