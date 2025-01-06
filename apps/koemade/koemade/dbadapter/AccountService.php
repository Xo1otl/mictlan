<?php

namespace koemade\dbadapter;

use koemade\admin;
use koemade\util;
use PDO;
use PDOException;

class AccountService implements admin\AccountService
{
    private PDO $conn;
    private util\Logger $logger;

    public function __construct()
    {
        $this->conn = DBConnection::getInstance();
        $this->logger = util\Logger::getInstance();
    }

    /**
     * @inheritDoc
     */
    public function banAccount(string $username): void
    {
        $this->logger->info("Banning account: $username");

        try {
            $stmt = $this->conn->prepare("UPDATE accounts SET status = 'banned' WHERE username = :username");
            $stmt->execute([':username' => $username]);

            if ($stmt->rowCount() > 0) {
                $this->logger->info("Account $username has been banned.");
            } else {
                $this->logger->warning("Account $username not found.");
            }
        } catch (PDOException $e) {
            $this->logger->error("Failed to ban account $username: " . $e->getMessage());
            throw $e;
        }
    }

    /**
     * @inheritDoc
     */
    public function createAccount(admin\CreateAccountInput $input): void
    {
        $this->logger->info("Creating account: {$input->username}");

        try {
            $this->conn->beginTransaction();

            // アカウントを accounts テーブルに挿入
            $stmt = $this->conn->prepare("INSERT INTO accounts (username, password) VALUES (:username, :password)");
            $stmt->execute([
                ':username' => $input->username,
                ':password' => password_hash($input->password, PASSWORD_BCRYPT)
            ]);

            // 新しく作成されたアカウントのIDを取得
            $accountId = $this->conn->lastInsertId();

            // ロールを account_role テーブルに挿入（ロール名を直接指定）
            foreach ($input->roles as $role) {
                $stmt = $this->conn->prepare("
                    INSERT INTO account_role (account_id, role_id)
                    SELECT :account_id, id
                    FROM roles
                    WHERE name = :role_name
                ");
                $stmt->execute([
                    ':account_id' => $accountId,
                    ':role_name' => $role
                ]);

                if ($stmt->rowCount() === 0) {
                    throw new PDOException("Role $role not found.");
                }
            }

            $this->conn->commit();
            $this->logger->info("Account " . $input->username . " created successfully.");
        } catch (PDOException $e) {
            $this->conn->rollBack();
            $this->logger->error("Failed to create account: " . $e->getMessage());
            throw $e;
        }
    }

    /**
     * @inheritDoc
     */
    public function deleteAccount(string $username): void
    {
        $this->logger->info("Deleting account: $username");

        try {
            $stmt = $this->conn->prepare("DELETE FROM accounts WHERE username = :username");
            $stmt->execute([':username' => $username]);

            if ($stmt->rowCount() > 0) {
                $this->logger->info("Account $username has been deleted.");
            } else {
                $this->logger->warning("Account $username not found.");
            }
        } catch (PDOException $e) {
            $this->logger->error("Failed to delete account $username: " . $e->getMessage());
            throw $e;
        }
    }
}
