<?php

require __DIR__ . '/middleware.php';

authenticateAdmin();

# 既に管理者として認証済みなので、postやgetのパラメータを信用する
getProfileController()->handleUpdateProfile($_POST, new \common\Id($_POST['id']));
