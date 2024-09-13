<?php

require __DIR__ . '/middleware.php';

$controller = getController();

$controller->handleEditPassword($_POST);

$repo = new \filesystem\SessionRepo();
$repo->delete();

echo "<p>パスワードを変更しました。</p>";
echo "<a href='signin.html'>こちらから新しいパスワードでログインしてください。</a>";
