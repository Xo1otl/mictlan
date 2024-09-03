<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    require __DIR__ . '/middleware.php';

    $username = new \auth\Username($_POST["username"]);
    $passwordText = $_POST["password"];
    $signUpInput = new \auth\SignInInput($username, $passwordText);

    $app = NewApp();
    try {
        $account = $app->signin($signUpInput);
    } catch (\Exception $e) {
        echo "" . $e->getMessage() . "";
    }
    if ($account->role->value === "admin") {
        echo "
        <h2>管理者としてログインしました</h2>
        <p><a href='./signup.php'>新規ユーザー登録ページへ</a></p>
        ";
    } else if ($account->role->value === "actor") {
        echo "
        <h2>ログインしました</h2>
        <p>ようこそ、" . htmlspecialchars($account->username) . "さん</p>
        ";
    }
    return;
}
?>

<!DOCTYPE html>
<html lang="ja">

<head>
    <meta charset="UTF-8">
    <title>ログイン</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f2f2f2;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }

        .signin-container {
            background-color: #fff;
            padding: 30px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            text-align: center;
        }

        h1 {
            margin-top: 0;
        }

        input[type="email"],
        input[type="password"] {
            width: 100%;
            padding: 12px 20px;
            margin: 8px 0;
            display: inline-block;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-sizing: border-box;
        }

        button {
            background-color: #4CAF50;
            color: white;
            padding: 14px 20px;
            margin: 8px 0;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            width: 100%;
        }
    </style>
</head>

<body>
    <div class="signin-container">
        <h1>ログイン</h1>
        <form action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]); ?>" method="post">
            <label for="username">メールアドレス</label>
            <input type="email" id="username" name="username" placeholder="メールアドレスを入力" required>

            <label for="password">パスワード</label>
            <input type="password" id="password" name="password" placeholder="パスワードを入力" required>

            <button type="submit">ログイン</button>
        </form>
    </div>
</body>

</html>