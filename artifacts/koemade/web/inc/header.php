<?php
require_once __DIR__ . '/../auth/middleware.php';

$session = getSession();
if ($session->role == \auth\Role::ADMIN) {
?>
<header>
    <nav>
        <ul>
            <li><a href="../auth/list_accounts.php">アカウント一覧</a></li>
            <li><a href="../auth/signup.php">アカウント作成</a></li>
            <li><a href="../auth/ban.php">アカウントBAN</a></li>
            <li><a href="../auth/signin.php?signout">ログアウト</a></li>
        </ul>
    </nav>
</header>
<?php
} else if ($session->role == \auth\Role::ACTOR) {
    $profile = getProfileController()->handleGetOrInitProfile($session->accountId);

?>
<header>
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
                <li><a href="../actor/profile.php">基本情報</a></li>
                <li><a href="../actor/profile.php?mode=avator">プロフィール画像</a></li>
                <li><a href="../actor/profile.php?mode=r-option">Rオプション設定</a></li>
            </ul>
            </li>
            <li><b>ボイスサンプル</b>
                <ul>
                    <li><a href="../actor/profile.php?mode=add-voice">新規追加</a></li>
                    <li><a href="../actor/list_voice.php">サンプル一覧</a></li>
                </ul>
            </li>
            <li><a href="../auth/signin.php?signout">ログアウト</a></li>
        </ul>
    </nav>
</header>
<?php
} 

