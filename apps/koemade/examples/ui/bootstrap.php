<?php
require_once __DIR__ . "/../../koemade/kernel/bootstrap.php";

// $apiURL = "http://localhost:8002/api";
$apiURL = "http://stg5.koemade.net";
?>

<!DOCTYPE html>
<html lang="ja">

<head>
    <meta charset="UTF-8">
    <title>System Interface</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }

        nav {
            background-color: #333;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }

        nav ul {
            list-style-type: none;
            margin: 0;
            padding: 0;
            display: flex;
            gap: 20px;
        }

        nav ul li {
            display: inline;
        }

        nav a {
            color: white;
            text-decoration: none;
            padding: 10px 15px;
            border-radius: 3px;
            transition: background-color 0.3s;
        }

        nav a:hover {
            background-color: #555;
        }

        #debug_msg_window {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #fff;
            border-top: 2px solid #ccc;
            padding: 15px;
            font-family: monospace;
            max-height: 200px;
            overflow-y: auto;
            box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.05);
        }
    </style>
</head>

<body>
    <nav>
        <ul>
            <li><a href="/guest/search-example.php">Search Voices/Actors</a></li>
            <li><a href="/guest/query-example.php">Query Voice/Actor</a></li>
            <li><a href="/guest/login-example.php">Guest Login</a></li>
            <li><a href="/guest/submit-example.php">Guest Submit</a></li>
            <li><a href="/actor/profile-example.php">Actor Profile</a></li>
            <li><a href="/actor/voice-example.php">Actor Voices</a></li>
            <li><a href="/admin-example.php">Admin</a></li>
        </ul>
    </nav>
</body>

</html>