<?php

namespace koemade\auth;

use koemade\common;
use PDO;

class PdoAccountRepo implements AccountRepo
{
    private PDO $pdo;

    public function __construct()
    {
        $this->pdo = common\MySql::connection();
    }

    public function add(SignUpInput $signUpInput): AccountId
    {
        $stmt = $this->pdo->prepare("INSERT INTO accounts (username, password) VALUES (:username, :password)");
        $stmt->execute([
            ':username' => $signUpInput->username,
            ':password' => (string) $signUpInput->password,
        ]);

        return new AccountId($this->pdo->lastInsertId());
    }

    public function editPassword(EditPasswordInput $input)
    {
        $stmt = $this->pdo->prepare("UPDATE accounts SET password = :new_password WHERE username = :username");
        $stmt->execute([
            ':new_password' => (string) $input->newPassword,
            ':username' => $input->username,
        ]);
    }

    public function deleteById(AccountId $id)
    {
        $stmt = $this->pdo->prepare("DELETE FROM accounts WHERE id = :id");
        $stmt->execute([':id' => $id]);
    }

    public function deleteByUsername(Username $username)
    {
        $stmt = $this->pdo->prepare("DELETE FROM accounts WHERE username = :username");
        $stmt->execute([':username' => $username]);
    }

    public function findByUsername(Username $username): Account
    {
        $stmt = $this->pdo->prepare("
            SELECT a.id, a.username, a.password, r.role_name
            FROM accounts a
            LEFT JOIN account_roles ar ON a.id = ar.account_id
            LEFT JOIN roles r ON ar.role_id = r.id
            WHERE a.username = :username
        ");
        $stmt->execute([':username' => $username]);
        $result = $stmt->fetch(PDO::FETCH_ASSOC);

        if (!$result) {
            throw new \Exception("Account not found");
        }

        return new Account(
            new AccountId($result['id']),
            new Username($result['username']),
            new Password($result['password'], false),
            new Role($result['role_name'] ?? Role::ACTOR)
        );
    }

    public function findById(AccountId $id): Account
    {
        $stmt = $this->pdo->prepare("
            SELECT a.id, a.username, a.password, r.role_name
            FROM accounts a
            LEFT JOIN account_roles ar ON a.id = ar.account_id
            LEFT JOIN roles r ON ar.role_id = r.id
            WHERE a.id = :id
        ");
        $stmt->execute([':id' => $id]);
        $result = $stmt->fetch(PDO::FETCH_ASSOC);

        if (!$result) {
            throw new \Exception("Account not found");
        }

        return new Account(
            new AccountId($result['id']),
            new Username($result['username']),
            new Password($result['password'], false),
            new Role($result['role_name'] ?? Role::ACTOR)
        );
    }

    public function allAccounts(): array
    {
        $stmt = $this->pdo->query("
            SELECT a.id, a.username, a.password, r.role_name
            FROM accounts a
            LEFT JOIN account_roles ar ON a.id = ar.account_id
            LEFT JOIN roles r ON ar.role_id = r.id
        ");
        $results = $stmt->fetchAll(PDO::FETCH_ASSOC);

        return array_map(function ($result) {
            return new Account(
                new AccountId($result['id']),
                new Username($result['username']),
                new Password($result['password'], false),
                new Role($result['role_name'] ?? Role::ACTOR)
            );
        }, $results);
    }
}