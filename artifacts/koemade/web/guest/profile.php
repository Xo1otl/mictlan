<?php

require __DIR__ . '/../../vendor/autoload.php';

// URLからaccount_idを取得する
if (isset($_GET['id'])) {
    $account_id = $_GET['id'];
} else {
    die('Account IDが指定されていません。');
    // echo "Account IDが指定されていません。";
    // exit();
}

$profileRepo = new \mysql\ProfileRepo();
$app = new \actor\App($profileRepo);
$controller = new \actor\Controller($app);
try {
    $profile = $controller->handleGetOrInitProfile(new \common\Id($account_id));
} catch (\Exception $e) {
    die("Profile not found");
    // echo "Profile not found";
    // exit();
}
?>

<!DOCTYPE html>
<html lang="ja">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>マイページ表示</title>
    <link rel="stylesheet" href="../assets/css/style.css">
    <link rel="stylesheet" href="../assets/css/fontawesome.min.css">
    <link rel="stylesheet" href="../assets/css/brands.min.css">
    <link rel="stylesheet" href="../assets/css/solid.min.css">
</head>

<body>
    <div class="profile show container">
        <h1>声優プロフィール</h1>
        <!-- General Profile Display -->
        <div>
            <label>表示名</label>
            <div class="field-value"><?php echo htmlspecialchars($profile->displayName, ENT_QUOTES, 'UTF-8'); ?></div>
        </div>
        <div>
            <label>声優ランク</label>
            <div class="field-value">
                <?php
                switch ($profile->category) {
                    case \actor\Category::PROFESSIONAL_VETERAN:
                        echo 'プロベテラン';
                        break;
                    case \actor\Category::PROFESSIONAL_ROOKIE:
                        echo 'プロルーキー';
                        break;
                    case \actor\Category::AMATEUR:
                        echo '新人・アマチュア';
                        break;
                    default:
                        echo '不明';
                }
                ?>
            </div>
        </div>
        <div>
            <label>自己PR</label>
            <div class="field-value">
                <?php echo nl2br(htmlspecialchars($profile->selfPromotion, ENT_QUOTES, 'UTF-8')); ?>
            </div>
        </div>
        <div>
            <label>価格</label>
            <div class="field-value"><?php echo htmlspecialchars($profile->price, ENT_QUOTES, 'UTF-8'); ?> 円/30分</div>
        </div>

        <!-- R-related Options Display -->
        <div>
            <label>セクシャル（R的要素）表現</label>
            <div class="field-value"><?php echo $profile->nsfwOptions->ok ? '許可' : '不可'; ?></div>
        </div>
        <div>
            <label>R作品</label>
            <div class="field-value"><?php echo htmlspecialchars($profile->nsfwOptions->price, ENT_QUOTES, 'UTF-8'); ?>
                円/30分
            </div>
        </div>
        <div>
            <label>R過激表現オプション</label>
            <div class="field-value"><?php echo $profile->nsfwOptions->hardOk ? '許可' : '不可'; ?></div>
        </div>
        <div>
            <label>過激オプション割増</label>
            <div class="field-value">
                <?php echo htmlspecialchars($profile->nsfwOptions->hardSurcharge, ENT_QUOTES, 'UTF-8'); ?>
            </div>
        </div>
    </div>
</body>

</html>