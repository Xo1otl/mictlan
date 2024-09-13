<?php

require __DIR__ . '/middleware.php';

$session = authenticate();
$message = '';
if ($session->role->value === \auth\Role::ADMIN) {
    $accountId = $_GET['id'];
    if (!$accountId) {
        header('Location: ../auth/list_accounts.php', true, 302);
        // echo "<a href='../auth/list_accounts.php'>アカウントを選択して編集</a>";
        // exit();
    }
    $message = '<p class="notice">管理者メッセージ: accountId ' . $accountId . 'を閲覧/編集しています。</p>';
    $session->accountId->value = $accountId;
    $sessionRepo = new \filesystem\SessionRepo();
    $sessionRepo->set($session);
}

$profile = getProfileController()->handleGetOrInitProfile($session->accountId);
?>

<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>マイページ編集</title>
    <link rel="stylesheet" href="../assets/css/style.css">
    <link rel="stylesheet" href="../assets/css/fontawesome.min.css">
    <link rel="stylesheet" href="../assets/css/brands.min.css">
    <link rel="stylesheet" href="../assets/css/solid.min.css">
</head>
<body class="mypage">
<?= $message; ?>
<?php require('../inc/header.php'); ?>
<div class="edit container">


    <?php if ( $_GET['mode'] === 'avator' ) : ?>
    <!-- Profile Image Section -->
    <h1>プロフィール画像</h1>
    <form action="update_thumbnail.php" method="post" enctype="multipart/form-data" id="profile_image_form">
        <div>
            <figure class="current-avator">
                <figcaption>現在のプロフィール画像</figcaption>
                <?php
                $profileImage = $profile->profileImage;

                if ($profileImage !== null) {
                    $imageSrc = "../../uploads/actor/" . htmlspecialchars($profileImage->getFullname());
                    $altText = "プロフィール画像";
                    echo '<img src="' . $imageSrc . '" alt="' . $altText . '" style="max-width: 256px;">';
                } else {
                    echo '<img src="../assets/images/user-icon__preset.png" alt="プロフィール画像" class="avator">';
                }
                ?>
            </figure>
            <input type="file" id="profile_image" name="profile_image" accept="image/*">
        </div>
        <div>
            <button type="submit" class="secondary">画像更新</button>
        </div>
    </form>
    <script>
        document.getElementById('profile_image_form').addEventListener('submit', function (event) {
            event.preventDefault();

            const formData = new FormData(this);

            fetch('update_thumbnail.php', {
                method: 'POST',
                body: formData
            })
                .then(response => response.text())
                .then(data => {
                    console.log(data);
                    window.location.assign('')
                })
                .catch(error => {
                    console.error('Error:', error);
                    // Handle any errors
                });
        });
    </script>

    <?php elseif ( $_GET['mode'] === 'r-option' ) : ?>
    <h1>Rオプション設定</h1>
    <!-- R-related Options Form -->
    <form action="update_r.php" method="post" id="r_options_form">
        <div>
            <label for="ok">セクシャル（R的要素）表現</label>
            <input type="checkbox" id="ok" name="ok" <?php echo $profile->r->ok ? 'checked' : ''; ?>>
        </div>
        <div>
            <label for="price">R作品</label>
            <input type="number" id="price" name="price"
                   value="<?php echo htmlspecialchars($profile->r->price, ENT_QUOTES, 'UTF-8'); ?>"> 円/30分
        </div>
        <div>
            <label for="hardOk">R過激表現オプション</label>
            <input type="checkbox" id="hardOk" name="hardOk" <?php echo $profile->r->hardOk ? 'checked' : ''; ?>>
        </div>
        <div>
            <label for="hardSurcharge">過激オプション割増</label>
            <input type="number" id="hardSurcharge" name="hardSurcharge"
                   value="<?php echo htmlspecialchars($profile->r->hardSurcharge, ENT_QUOTES, 'UTF-8'); ?>">
        </div>
        <div>
            <button type="submit" class="secondary">Rオプション更新</button>
        </div>
    </form>
    <script>
        document.getElementById('r_options_form').addEventListener('submit', function (event) {
            event.preventDefault();

            const formData = new FormData(this);

            fetch('update_r.php', {
                method: 'POST',
                body: formData
            })
                .then(response => response.text())
                .then(data => {
                    console.log(data);
                    // Handle the response from the server
                    window.alert("更新しました")
                })
                .catch(error => {
                    console.error('Error:', error);
                    // Handle any errors
                });
        });
    </script>


    <?php elseif ( $_GET['mode'] === 'add-voice' ) : ?>
    <h1>音声投稿</h1>
    <form action="upload_voice.php" method="post" enctype="multipart/form-data" id="voice_form">
        <div>
            <label for="voiceTitle">音声タイトル</label>
            <input type="text" id="voiceTitle" name="voiceTitle">
        </div>
        <label for="audio">音声ファイルを選択 (mp3, wav, ogg):</label>
        <input type="file" id="audio" name="audio" accept=".mp3, .wav, .ogg" required>
        <input type="submit" class="secondary" value="アップロード">
    </form>
    <script>
        document.getElementById('voice_form').addEventListener('submit', function (event) {
            event.preventDefault();

            const formData = new FormData(this);

            fetch('upload_voice.php', {
                method: 'POST',
                body: formData
            })
                .then(response => response.text())
                .then(data => {
                    console.log(data);
                    // Handle the response from the server
                    window.alert("更新しました")
                })
                .catch(error => {
                    console.error('Error:', error);
                    // Handle any errors
                });
        });
    </script>


    <?php else : ?>
    <!-- General Profile Form -->
    <h1>プロフィール設定</h1>
    <form action="update_profile.php" method="post" id="profile_form">
        <div>
            <label for="displayName">表示名</label>
            <input type="text" id="displayName" name="displayName"
                   value="<?php echo htmlspecialchars($profile->displayName, ENT_QUOTES, 'UTF-8'); ?>">
        </div>
        <div>
            <label>声優ランク</label>
            <div class="radio-group">
                <label for="professionalVeteran">
                    <input type="radio" id="professionalVeteran" name="category"
                           value="<?php echo \actor\Category::PROFESSIONAL_VETERAN; ?>" <?php echo \actor\Category::PROFESSIONAL_VETERAN === $profile->category ? 'checked' : ''; ?>>
                    プロベテラン
                </label>
                <label for="professionalRookie">
                    <input type="radio" id="professionalRookie" name="category"
                           value="<?php echo \actor\Category::PROFESSIONAL_ROOKIE; ?>" <?php echo \actor\Category::PROFESSIONAL_ROOKIE === $profile->category ? 'checked' : ''; ?>>
                    プロルーキー
                </label>
                <label for="amateur">
                    <input type="radio" id="amateur" name="category"
                           value="<?php echo \actor\Category::AMATEUR; ?>" <?php echo \actor\Category::AMATEUR === $profile->category ? 'checked' : ''; ?>>
                    新人・アマチュア
                </label>
            </div>
        </div>
        <div>
            <label for="selfPromotion">自己PR</label>
            <textarea id="selfPromotion"
                      name="selfPromotion"><?php echo htmlspecialchars($profile->selfPromotion, ENT_QUOTES, 'UTF-8'); ?></textarea>
        </div>
        <div>
            <label for="price">価格</label>
            <input type="number" id="price" name="price"
                   value="<?php echo htmlspecialchars($profile->price, ENT_QUOTES, 'UTF-8'); ?>">
            円/30分
        </div>
        <div>
            <button type="submit" class="secondary">プロフィール更新</button>
        </div>
    </form>
    <script>
        document.getElementById('profile_form').addEventListener('submit', function (event) {
            event.preventDefault();

            const formData = new FormData(this);

            fetch('update_profile.php', {
                method: 'POST',
                body: formData
            })
                .then(response => response.text())
                .then(data => {
                    console.log(data);
                    // Handle the response from the server
                    window.alert("更新しました")
                })
                .catch(error => {
                    console.error('Error:', error);
                    // Handle any errors
                });
        });
    </script>
    <?php endif; ?>


</div>
</body>
</html>
