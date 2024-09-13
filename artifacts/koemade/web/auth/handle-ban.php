<?php

require __DIR__ . '/middleware.php';

$session = getSession();

if ($session->role->value === \auth\Role::ADMIN) {
    $controller = getController();
    $controller->handleDeleteAccountByAdmin($_POST);
    $message = '管理者としてアカウントを削除しました<br><a class="button" href="list_accounts.php">アカウント一覧</a>';
    // exit();
} else {

    $controller = getController();
    $controller->handleDeleteAccount($_POST);
    
    $sessionRepo = new \filesystem\SessionRepo();
    $sessionRepo->delete();
    $message = "アカウントを削除しました";
        
}

?>
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title></title>
    <link rel="stylesheet" href="../assets/css/style.css">
    <link rel="stylesheet" href="../assets/css/fontawesome.min.css">
    <link rel="stylesheet" href="../assets/css/brands.min.css">
    <link rel="stylesheet" href="../assets/css/solid.min.css">
</head>
<body>
<?php require('../inc/header.php'); ?>
<div class="container form edit-password">
    <?= $message; ?>
</div>
</body>
</html>
