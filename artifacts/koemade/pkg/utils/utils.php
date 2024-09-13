<?php


namespace utils;

function generateUUID(): ?string
{
    try {
        $data = random_bytes(16);
    } catch (\Exception $e) {
        return null; // Handle the exception or log it as needed
    }

    assert(strlen($data) == 16);

    $data[6] = chr((ord($data[6]) & 0x0f) | 0x40); // set version to 0100
    $data[8] = chr((ord($data[8]) & 0x3f) | 0x80); // set bits 6-7 to 10

    return sprintf('%s-%s-%s-%s-%s',
        bin2hex(substr($data, 0, 4)),
        bin2hex(substr($data, 4, 2)),
        bin2hex(substr($data, 6, 2)),
        bin2hex(substr($data, 8, 2)),
        bin2hex(substr($data, 10, 6))
    );
}


/**
 * 条件を満たすランダムなパスワードを生成する関数
 *
 * @param int $length パスワードの長さ（デフォルトは8）
 * @return string 生成されたパスワード
 * @throws \Random\RandomException
 */
function generatePlainPassword(int $length = 8): string
{
    if ($length < 8) {
        throw new \InvalidArgumentException('Password length must be at least 8 characters.');
    }

    $letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';
    $digits = '0123456789';
    $specialCharacters = '!@#$%^&*()';
    $characters = $letters . $digits . $specialCharacters;

    $password = $letters[random_int(0, strlen($letters) - 1)];
    $password .= $digits[random_int(0, strlen($digits) - 1)];
    $password .= $specialCharacters[random_int(0, strlen($specialCharacters) - 1)];

    // 残りの文字を追加
    for ($i = 2; $i < $length; $i++) {
        $password .= $characters[random_int(0, strlen($characters) - 1)];
    }

    // パスワードをシャッフル
    return str_shuffle($password);
}
