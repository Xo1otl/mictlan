<?php

namespace koemade\common;

use PDO;
use PDOException;

class MySql
{
    private static ?PDO $connection = null;

    public static function connection(): PDO
    {
        $config = include __DIR__ . "/config.php";
        $host = $config['db_host'];
        $user = $config['db_user'];
        $password = $config['db_password'];
        $database = $config['db_database'];
        $charset = 'utf8mb4';

        $dsn = "mysql:host=$host;dbname=$database;charset=$charset";
        $options = [
            PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
            PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
            PDO::ATTR_EMULATE_PREPARES => false,
        ];

        if (self::$connection === null) {
            try {
                self::$connection = new PDO($dsn, $user, $password, $options);
            } catch (PDOException $e) {
                die("Connection failed: " . $e->getMessage());
            }
        } else {
            // Check if the connection is alive
            try {
                self::$connection->query('SELECT 1');
            } catch (PDOException $e) {
                // Connection is not alive, create a new one
                try {
                    self::$connection = new PDO($dsn, $user, $password, $options);
                } catch (PDOException $e) {
                    die("Connection failed: " . $e->getMessage());
                }
            }
        }

        return self::$connection;
    }
}