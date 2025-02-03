<?php

require_once __DIR__ . '/../bootstrap.php';

use koemade\guest;

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // 画像アップロード処理
    if (isset($_FILES['image']) && $_FILES['image']['error'] === UPLOAD_ERR_OK) {
        // 一時ファイルのパスを取得
        $tmpFilePath = $_FILES['image']['tmp_name'];
        $imagePath = $tmpFilePath; // 一時ファイルのパスをそのまま使用
    } else {
        die("画像がアップロードされていないか、エラーが発生しました。");
    }

    // フォームからのデータを受け取る
    $email = $_POST['email'];
    $name = $_POST['name'];
    $kana = $_POST['kana'];
    $address = $_POST['address'];
    $phone = $_POST['phone'];
    $bankName = $_POST['bank_name'];
    $branchName = $_POST['branch_name'];
    $accountNumber = $_POST['account_number'];
    $message = $_POST['message'];

    // オブジェクトを作成
    $japaneseName = new guest\JapaneseName($name, $kana);
    $beneficiaryInfo = new guest\BeneficiaryInfo($bankName, $branchName, $accountNumber);

    $signupRequest = new guest\SignupRequest(
        $email,
        $japaneseName,
        $address,
        $phone,
        $imagePath, // 一時ファイルのパスをそのまま渡す
        $beneficiaryInfo,
        $message
    );

    $service = new guest\MailService();
    $service->notify($signupRequest);
} else {
    // フォームを表示
    ?>
    <!DOCTYPE html>
    <html lang="ja">

    <head>
        <meta charset="UTF-8">
        <title>会員登録フォーム</title>
        <script>
            function submitForm(event) {
                event.preventDefault(); // フォームのデフォルト送信を防止

                const formData = new FormData(document.getElementById('signup-form'));

                // リクエストを送信（レスポンスは待たない）
                fetch('', {
                    method: 'POST',
                    body: formData,
                });

                document.getElementById('signup-div').style.display = 'none'; // フォームを非表示
                document.getElementById('thank-you-message').style.display = 'block'; // メッセージを表示
            }
        </script>
        <style>
            #thank-you-message {
                display: none;
                /* 初期状態では非表示 */
            }
        </style>
    </head>

    <body>
        <div id="signup-div">
            <h1>会員登録フォーム</h1>
            <form id="signup-form" method="POST" action="" enctype="multipart/form-data" onsubmit="submitForm(event)">
                <label for="email">メールアドレス:</label>
                <input type="email" id="email" name="email" required><br>

                <label for="name">名前:</label>
                <input type="text" id="name" name="name" required><br>

                <label for="kana">カナ:</label>
                <input type="text" id="kana" name="kana" required><br>

                <label for="address">住所:</label>
                <input type="text" id="address" name="address" required><br>

                <label for="phone">電話番号:</label>
                <input type="text" id="phone" name="phone" required><br>

                <label for="image">画像:</label>
                <input type="file" id="image" name="image" required><br>

                <label for="bank_name">銀行名:</label>
                <input type="text" id="bank_name" name="bank_name" required><br>

                <label for="branch_name">支店名:</label>
                <input type="text" id="branch_name" name="branch_name" required><br>

                <label for="account_number">口座番号:</label>
                <input type="text" id="account_number" name="account_number" required><br>

                <label for="message">メッセージ:</label>
                <textarea id="message" name="message" required></textarea><br>

                <input type="hidden" name="csrf_token" value="<?php echo $_SESSION['csrf_token'] ?>"> <!-- CSRFトークンを埋め込む -->
                <button type="submit">登録</button>
            </form>
        </div>
        <div id="thank-you-message">
            <h1>お申し込みありがとうございます</h1>
            <p>登録申し込みの受け付けが完了しました。<br>
                入力いただいたメールアドレス宛てに確認メールを2〜3日以内に送信しますので、<br>
                記載されている手順に従って登録を完了してください。</p>
        </div>
    </body>

    </html>
    <?php
}
