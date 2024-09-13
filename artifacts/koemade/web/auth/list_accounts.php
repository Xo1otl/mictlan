<?php

require __DIR__ . '/middleware.php';

$session = getSession();

if ($session->role != \auth\Role::ADMIN) {
    http_response_code(400);
    exit('Access Denied');
    // echo "Access Denied";
    // exit();
}

$controller = getController();
$accounts = $controller->handleGetAllAccounts();
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
<body class="mypage">
<?php require('../inc/header.php'); ?>
<div class="container voice-list">
    <h1>アカウント一覧</h1>
    <table class="accounts">
        <thead>
            <tr>
                <th>ID</th>
                <th>ユーザー名</th>
            </tr>
        </thead>
        <tbody>
<?php

foreach ($accounts as $account) {
    echo '<tr><td>' . $account->id . '</td><td><a href="../actor/profile.php?id=' . $account->id . '">' . $account->username . '</a></td></tr>';
}
?>
        </tbody>
    </table>


</div>
</body>
</html>

