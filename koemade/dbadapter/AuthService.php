<?php

namespace koemade\dbadapter;

use koemade\auth;
use PDO;
use PDOException;
use Exception;
use Firebase\JWT\JWT;

class AuthService implements auth\AuthService
{
    private PDO $conn;
    private string $jwtSecretKey;

    public function __construct(PDO $conn, string $jwtSecretKey)
    {
        $this->conn = $conn;
        $this->jwtSecretKey = $jwtSecretKey;
    }

    /**
     * アカウントを認証し、JWT トークンを返す
     *
     * @param string $username アカウントのユーザー名
     * @param string $password アカウントのパスワード
     * @return string 認証成功時に JWT トークンを返す
     * @throws Exception 認証失敗時またはデータベースエラー時に例外をスロー
     */
    public function authenticate(string $username, string $password): string
    {
        try {
            $stmt = $this->conn->prepare("
                SELECT a.id, a.password, a.status, r.name AS role
                FROM accounts a
                LEFT JOIN account_role ar ON a.id = ar.account_id
                LEFT JOIN roles r ON ar.role_id = r.id
                WHERE a.username = :username
            ");
            $stmt->execute(['username' => $username]);
            $account = $stmt->fetch(PDO::FETCH_ASSOC);

            if (!$account) {
                throw new exceptions\AuthenticationException("ユーザー名またはパスワードが正しくありません。");
            }

            if (password_verify($password, $account['password'])) {
                return $this->generateJwtToken($account);
            } else {
                throw new Exception("ユーザー名またはパスワードが正しくありません。");
            }
        } catch (PDOException $e) {
            throw new exceptions\DatabaseException("データベースエラーが発生しました。");
        }
    }

    /**
     * JWT トークンを生成する
     *
     * @param array{
     *     id: int,
     *     role: string,
     *     status: string
     * } $account アカウント情報
     * @return string 生成された JWT トークン
     */
    private function generateJwtToken(array $account): string
    {
        $now = time();
        $payload = [
            'sub' => $account['id'],
            'exp' => $now + 3600,
            'iat' => $now,
            'nbf' => $now,
            'role' => $account['role'],
            'status' => $account['status']
        ];

        return JWT::encode($payload, $this->jwtSecretKey, 'HS256');
    }
}
