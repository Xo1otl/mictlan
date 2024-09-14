<?php

require_once __DIR__ . '/middleware.php';

try {
    if (!isset($_GET['id']) || !is_numeric($_GET['id'])) {
        throw new Exception('Invalid id');
    }
    $id = new \common\Id($_GET['id']);
    # sessionではなく、クエリパラメータを使ってprofileオブジェクトを取得
    $profile = getProfileController()->handleGetOrInitProfile($id);
} catch (Exception $e) {
    $message = '<p class="error">プロフィールの取得に失敗しました</p>';
}

$profile = getProfileController()->handleGetOrInitProfile($id);
?>

<header>
    <nav>
        <ul>
            <li><a href="list_accounts.php">アカウント一覧</a></li>
            <li><a href="../auth/signup.php">アカウント作成</a></li>
            <li><a href="../auth/ban.php">アカウントBAN</a></li>
            <li><a href="../auth/signin.php?signout">ログアウト</a></li>
        </ul>
    </nav>
    <figure class="avator">
        <?php
            $profileImage = $profile->profileImage;

            if ($profileImage !== null) {
                $imageSrc = "../../uploads/actor/" . htmlspecialchars($profileImage->getFullname());
                $altText = "プロフィール画像";
                echo '<img src="' . $imageSrc . '" alt="' . $altText . '">';
            } else {
                echo '<img src="../assets/images/user-icon__preset.png" alt="プロフィール画像" class="avator">';
            }
            ?>
        <figcaption>
            <?php echo htmlspecialchars($profile->displayName, ENT_QUOTES, 'UTF-8'); ?>
        </figcaption>
    </figure>
    <nav>
        <ul>
            <li><b>マイページ</b>
            <ul>
                <li><a href="<?php echo "actor_profile.php?id=" . $id;?>">基本情報</a></li>
                <li><a href="<?php echo "actor_profile.php?mode=avator&id=" . $id;?>">プロフィール画像</a></li>
                <li><a href="<?php echo "actor_profile.php?mode=r-option&id=" . $id;?>">Rオプション設定</a></li>
            </ul>
            </li>
            <li><b>ボイスサンプル</b>
                <ul>
                    <li><a href="<?php echo "actor_profile.php?mode=add-voice&id=" . $id;?>">新規追加</a></li>
                    <li><a href="<?php echo "actor_list_voice.php?id=" . $id;?>">サンプル一覧</a></li>
                </ul>
            </li>
            <li><a href="../auth/signin.php?signout">ログアウト</a></li>
        </ul>
    </nav>
</header>
