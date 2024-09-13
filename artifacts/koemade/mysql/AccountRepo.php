<?php

namespace mysql;

use auth\EditPasswordInput;
use auth\Username;

class AccountRepo implements \auth\AccountRepo
{
    private \mysqli $mysqli;

    public function __construct()
    {
        $this->mysqli = DbConnection::getConnection();
    }

    /**
     * @throws \Exception
     */
    public function add(\auth\CredentialInput $credentialInput): \common\Id
    {
        $this->mysqli->begin_transaction();

        try {
            // Insert into accounts table
            $stmt = $this->mysqli->prepare("INSERT INTO accounts (username, password) VALUES (?, ?)");
            $hashedPassword = password_hash($credentialInput->password->text, PASSWORD_BCRYPT);

            $stmt->bind_param('ss', $credentialInput->username->text, $hashedPassword);
            $stmt->execute();
            $id = $stmt->insert_id;
            $stmt->close();

            // Insert default role into roles table if not exists
            $defaultRole = 'actor';
            $role_stmt = $this->mysqli->prepare("INSERT INTO roles (role_name) VALUES (?) ON DUPLICATE KEY UPDATE id=id");
            $role_stmt->bind_param('s', $defaultRole);
            $role_stmt->execute();
            $role_stmt->close();

            // Get the role id
            $role_id_stmt = $this->mysqli->prepare("SELECT id FROM roles WHERE role_name = ?");
            $role_id_stmt->bind_param('s', $defaultRole);
            $role_id_stmt->execute();
            $role_id_stmt->bind_result($role_id);
            $role_id_stmt->fetch();
            $role_id_stmt->close();

            // Insert into account_roles table
            $group_stmt = $this->mysqli->prepare("INSERT INTO account_roles (account_id, role_id) VALUES (?, ?)");
            $group_stmt->bind_param('ii', $id, $role_id);
            $group_stmt->execute();
            $group_stmt->close();

            // Commit transaction
            $this->mysqli->commit();

            return new \common\Id($id);
        } catch (\Exception $e) {
            // Rollback transaction on failure
            $this->mysqli->rollback();
            throw $e;
        }
    }

    /**
     * @throws \Exception
     */
    public function findByUsername(\auth\Username $username): \auth\Account
    {
        $stmt = $this->mysqli->prepare("SELECT a.id, a.username, a.password, r.role_name
                                FROM accounts a
                                LEFT JOIN account_roles ar ON a.id = ar.account_id
                                LEFT JOIN roles r ON ar.role_id = r.id
                                WHERE a.username = ?");
        $stmt->bind_param('s', $username->text);
        $stmt->execute();
        $stmt->bind_result($id, $username->text, $password, $group_name);
        $stmt->fetch();

        if ($id === null) {
            throw new \Exception("Account not found");
        }

        $account = new \auth\Account(new \common\Id((int)$id), $username, new \auth\PasswordHash($password), new \auth\Role($group_name));
        $stmt->close();

        return $account;
    }

    public function findById(\common\Id $id): \auth\Account
    {
        $stmt = $this->mysqli->prepare("SELECT a.id, a.username, a.password, r.role_name
                                FROM accounts a
                                LEFT JOIN account_roles ar ON a.id = ar.account_id
                                LEFT JOIN roles r ON ar.role_id = r.id
                                WHERE a.username = ?");
        $stmt->bind_param('i', $id->value);
        $stmt->execute();
        $stmt->bind_result($id->value, $username, $password, $group_name);
        $stmt->fetch();

        $account = new \auth\Account($id, new \auth\Username($username), new \auth\PasswordHash($password), new \auth\Role($group_name));
        $stmt->close();

        return $account;
    }

    public function deleteById(\common\Id $id)
    {
        \logger\imp("delete account ", $id);
        $stmt = $this->mysqli->prepare("DELETE FROM accounts WHERE id = ?");
        $stmt->bind_param('i', $id->value);
        $stmt->execute();
        $stmt->close();
    }

    public function deleteByUsername(Username $username)
    {
        \logger\imp("delete account ", $username);
        $stmt = $this->mysqli->prepare("DELETE FROM accounts WHERE username = ?");
        $stmt->bind_param('s', $username->text);
        $stmt->execute();
        $stmt->close();
    }

    /**
     * @throws \Exception
     */
    public function editPassword(EditPasswordInput $input)
    {
        $mysqli = $this->mysqli;

        // Start transaction
        $mysqli->begin_transaction();

        try {
            // Prepare and execute select statement to verify current credentials
            $stmt = $mysqli->prepare('SELECT password FROM accounts WHERE username = ?');
            $stmt->bind_param('s', $input->username->text);
            $stmt->execute();
            $stmt->store_result();

            if ($stmt->num_rows === 0) {
                throw new \Exception('User not found');
            }

            $stmt->bind_result($storedPassword);
            $stmt->fetch();

            if (!password_verify($input->password->text, $storedPassword)) {
                throw new \Exception('Incorrect current password');
            }

            $stmt->close();

            // Prepare and execute update statement to change the password
            $newPasswordHash = password_hash($input->newPassword->text, PASSWORD_BCRYPT);
            $stmt = $mysqli->prepare('UPDATE accounts SET password = ? WHERE username = ?');
            $stmt->bind_param('ss', $newPasswordHash, $input->username->text);
            $stmt->execute();

            // Commit transaction
            $mysqli->commit();
        } catch (\Exception $e) {
            // Rollback transaction on error
            $mysqli->rollback();
            \logger\err('Password change failed: ' . $e->getMessage());
            throw $e;
        }
    }

    /**
     * @return \auth\Account[]
     */
    public function getAllAccounts(): array
    {
        $mysqli = $this->mysqli;
        $accounts = [];

        $query = "SELECT id, username, password FROM accounts";
        if ($result = $mysqli->query($query)) {
            while ($row = $result->fetch_assoc()) {
                $id = new \common\Id($row['id']);
                $username = new \auth\Username($row['username']);
                $passwordHash = new \auth\PasswordHash($row['password']);
                $accounts[] = new \auth\Account($id, $username, $passwordHash);
            }
            $result->free();
        }

        return $accounts;
    }
}
