<?php

namespace mysql;

class DbConnection
{
    private static ?\mysqli $mysqli = null;

    public static function getConnection(): \mysqli
    {
        $config = include __DIR__ . "/config.php";
        $host = $config['db_host'];
        $user = $config['db_user'];
        $password = $config['db_password'];
        $database = $config['db_database'];

        if (self::$mysqli === null) {
            self::$mysqli = new \mysqli($host, $user, $password, $database);

            if (self::$mysqli->connect_error) {
                die("Connection failed: " . self::$mysqli->connect_error);
            }
        } else {
            // Check if the connection is alive
            if (!self::$mysqli->ping()) {
                // Close the current connection
                self::$mysqli->close();
                // Reinitialize the connection
                self::$mysqli = new \mysqli($host, $user, $password, $database);

                if (self::$mysqli->connect_error) {
                    die("Connection failed: " . self::$mysqli->connect_error);
                }
            }
        }
        return self::$mysqli;
    }
}
