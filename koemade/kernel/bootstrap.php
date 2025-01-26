<?php

require_once __DIR__ . '/../../vendor/autoload.php';
require_once __DIR__ . '/secrets.php';

use koemade\util;
use koemade\dbadapter;
use koemade\auth;
use koemade\storage;

// ロガーの初期化
$logger = util\Logger::getInstance();
$logger->info('Bootstraping the application');

// 認証サービスの初期化
$authService = new dbadapter\AuthService($secretKey);
$tokenService = new auth\JWTService($secretKey);

// ストレージの初期化
$storage = new storage\Storage();

// セッションを開始
session_start();
// echo session_id() . '<br>';
// echo 'request token: ' . $_POST['csrf_token'] . '<br>';
// echo 'session csrf_token: ' . $_SESSION['csrf_token'] . '<br>';
// if (!isset($_SESSION['request_count'])) {
//     $_SESSION['request_count'] = 0;
// }
// $_SESSION['request_count']++;
// echo 'Request Count: ' . $_SESSION['request_count'] . '<br>';

// CSRFトークンの検証処理
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // リクエストからCSRFトークンを取得
    $requestToken = $_POST['csrf_token'] ?? '';

    // セッションに保存されたCSRFトークンと比較
    if (empty($_SESSION['csrf_token']) || !hash_equals($_SESSION['csrf_token'], $requestToken)) {
        // トークンが不正な場合、エラーを返す
        http_response_code(403);
        $logger->error('Invalid CSRF token');
        echo json_encode(['error' => 'Invalid CSRF token']);
        exit;
    }

    // トークンが正しい場合、セッションのトークンを削除（1回限りの使用）
    unset($_SESSION['csrf_token']);
}

// CSRFトークンを生成（トークンが未設定の場合）
// GETリクエスト時に再生成すると、マルチタブや複数回処理の発生でトークンが一致しなくなるのでしない
if (empty($_SESSION['csrf_token'])) {
    $_SESSION['csrf_token'] = bin2hex(random_bytes(32)); // ランダムなトークンを生成
}
