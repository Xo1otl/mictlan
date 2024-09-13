<?php

require __DIR__ . '/middleware.php';

$controller = getController();
$controller->handleSignin($_POST);

$session = getSession();
if ($session->role == \auth\Role::ADMIN) {
    header('Location: ../actor/profile.php', true, 302);
    // echo '<a href="signup.php">管理者はこちらからアカウントを作成できます</a>';
} else if ($session->role == \auth\Role::ACTOR) {
    header('Location: ../actor/profile.php', true, 302);
    // echo '<a href="../actor/profile.php">マイページはこちら</a>';
} else {
    header('Location: ./signin.php?err=login-failed', true, 303);
    // echo "ログインに失敗しました";
}
