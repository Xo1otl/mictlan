<?php

require __DIR__ . '/middleware.php';

$authController = getController();

try {
    $id = $authController->handleSignup($_POST);
    header('Location: ../actor/profile.php?complete&id=' . $id, true, 302);
    // echo "ID " . $id . 'のアカウントを作成しました';
    // echo "<a href='../guest/profile.php?id=$id'>プロフィール画面を表示</a>";
} catch (\Exception $e) {
    \logger\err($e->getMessage());
    header('Location: ../auth/signup.php?err', true, 307);
}
