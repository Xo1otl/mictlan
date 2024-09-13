<?php

require __DIR__ . '/middleware.php';

$session = authenticate();

if ($session->role != \auth\Role::ACTOR && $session->role != \auth\Role::ADMIN) {
    header('Location: ../auth/signin.php', true, 302);
    // echo "<a href='../auth/signin.html'>声優の方はこちらからログインしてください。</a>";
    exit();
}

$allVoices = getVoiceController()->getAll($session->accountId);


/**
 * @param string $terms
 * @param string $name
 * 
 * @return string
 */
function get_tag( $terms, $name ) {
    foreach ($terms as $tag) {
        if ($tag->category === $name ) {
            return $tag->name;
        }
    }
}


?>
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title></title>
    <link rel="stylesheet" href="../assets/css/style.css">
    <link rel="stylesheet" href="../assets/css/fontawesome.min.css">
    <link rel="stylesheet" href="../assets/css/brands.min.css">
    <link rel="stylesheet" href="../assets/css/solid.min.css">
</head>
<body class="mypage">
<?php require('../inc/header.php'); ?>
<div class="container voice-list">
    <?php if ( $_GET['mode'] === 'edit' && preg_match( '/^\d+$/', $_GET['id'] ) ) : ?>
    <?php 
    $voice                  = $allVoices[$_GET['id']];
    $selected_age_tag       = get_tag($voice->tags, '年代別タグ' );
    $selected_character_tag = get_tag($voice->tags, 'キャラ別タグ' );
    ?>
    <h1>タイトル・タグ編集</h1>
    <div class="item">
        <audio controls src="../../uploads/voices/<?= htmlspecialchars($voice->getFullname()) ?>"></audio>
        <form method="POST" action="edit_tag.php">
            <input type="hidden" id="voice_id" name="voice_id" value="<?= htmlspecialchars($voice->voiceId->value) ?>"><br>
            <label for="voice_title">Voice Title:</label>
            <input class="width-100" type="text" id="voice_title" name="voice_title" value="<?= htmlspecialchars($voice->title) ?>"><br>
            <label for="age_tag">Age Tag:</label>
            <select class="width-100" id="age_tag" name="age_tag">
                <option value="" <?= ( ! $selected_age_tag ? ' selected' : '') ?>>選択する</option>
                <option value="10代"<?= ($selected_age_tag === '10代' ? ' selected' : '') ?>>10代</option>
                <option value="20代"<?= ($selected_age_tag === '20代' ? ' selected' : '') ?>>20代</option>
                <!-- <option value="30代以上"<?= ($selected_age_tag === '30代以上' ? ' selected' : '') ?>>30代以上</option> -->
                <option value="30代"<?= ($selected_age_tag === '30代' ? ' selected' : '') ?>>30代</option>
                <option value="40代以上"<?= ($selected_age_tag === '40代以上' ? ' selected' : '') ?>>40代以上</option>
            </select><br>
            <label for="character_tag">Character Tag:</label>
            <select class="width-100" id="character_tag" name="character_tag">
                <option value="" <?= ( ! $selected_character_tag ? ' selected' : '') ?>>選択する</option>
                <option value="大人しい"<?= ($selected_character_tag === '大人しい' ? ' selected' : '') ?>>大人しい</option>
                <option value="快活"<?= ($selected_character_tag === '快活' ? ' selected' : '') ?>>快活</option>
                <option value="セクシー・渋め"<?= ($selected_character_tag === 'セクシー・渋め' ? ' selected' : '') ?>>セクシー・渋め</option>
                <option value="幼い"<?= ($selected_character_tag === '幼い' ? ' selected' : '') ?>>幼い</option>
                <option value="真面目"<?= ($selected_character_tag === '真面目' ? ' selected' : '') ?>>真面目</option>
                <option value="その他"<?= ($selected_character_tag === 'その他' ? ' selected' : '') ?>>その他</option>
            </select><br>
            <button class="secondary width-100" type="submit" disabled="disabled"><i class="fa-solid fa-rotate"></i> 更新</button>
        </form>
        <form method="POST" action="delete_voice.php">
            <input type="hidden" id="voice_id" name="voice_id" value="<?= htmlspecialchars($voice->voiceId->value) ?>">
            <button class="width-100" type="submit"><i class="fa-solid fa-trash"></i> 削除</button>
        </form>
        <ul class="notes">
            <li>音声データの差し替えを行いたい場合は、いったん削除したうえで再アップロードをお願いします。</li>
            <li>削除しますとデータの復旧は出来ませんのでご注意下さい。</li>
        </ul>
    </div>


    <script>
        // Function to check if both selects have valid options selected
        function checkFormValidity() {
            const age_tag = document.getElementById('age_tag');
            const character_tag = document.getElementById('character_tag');
            const submitButton = document.querySelector('button[type="submit"]');

            // Check if both selects have a valid value selected
            const isSelect1Valid = age_tag.value === '';
            const isSelect2Valid = character_tag.value === '';
            console.log( (isSelect1Valid || isSelect2Valid) );

            // Enable or disable the submit button based on the validity of both selects
            submitButton.disabled = (isSelect1Valid || isSelect2Valid);
        }

        // Add event listeners to both selects
        window.onload = function() {
            document.getElementById('age_tag').addEventListener('change', checkFormValidity);
            document.getElementById('character_tag').addEventListener('change', checkFormValidity);
        };
    </script>
    <?php else : ?>


    <h1>音声一覧</h1>


    <div class="items">

        <?php
        foreach ($allVoices as $id => $voice) {

            $selected_age_tag       = get_tag($voice->tags, '年代別タグ' );
            $selected_character_tag = get_tag($voice->tags, 'キャラ別タグ' );

        ?>
        <div class="item">
            <h2><a href="../../uploads/voices/<?= htmlspecialchars($voice->getFullname()) ?>"><?= htmlspecialchars($voice->title) ?></a></h2>
            <audio controls src="../../uploads/voices/<?= htmlspecialchars($voice->getFullname()) ?>"></audio>
            <div class="tag age_tag">
                <span class="label">年代別タグ</span>
                <?= ( $selected_age_tag ? '<b><i class="fa-solid fa-tag"></i> ' . $selected_age_tag . '</b>' : '<b class="error"><i class="fa-solid fa-circle-exclamation"></i> タグを登録してください</b>' ) ?>
            </div>
            <div class="tag character_tag">
                <span class="label">キャラ別タグ</span>
                <?= ( $selected_character_tag ? '<b><i class="fa-solid fa-tag"></i> ' . $selected_character_tag . '</b>' : '<b class="error"><i class="fa-solid fa-circle-exclamation"></i> タグを登録してください</b>' ) ?>
            </div>
            <div class="edit">
                <a href="./list_voice.php?mode=edit&id=<?= $id ?>" class="button secondary"><i class="fa-solid fa-pen"></i> 編集</a>
            </div>
        </div>
<?php
    // echo '<div><a href="' . "../../uploads/voices/" . htmlspecialchars($voice->getFullname()) . '">' . htmlspecialchars($voice->title) . '</a></div>';
    // echo '<div>filename: ' . htmlspecialchars($voice->filename) . '</div>';
    // echo '<div>mimeType: ' . htmlspecialchars($voice->mimeType) . '</div>';
    // echo '<div>createdAt: ' . htmlspecialchars($voice->createdAt->format('Y-m-d')) . '</div>';

    // // Form to edit the tag of the voice
    // echo '<form method="POST" action="edit_tag.php">';
    // echo '<input type="hidden" id="voice_id" name="voice_id" value="' . htmlspecialchars($voice->voiceId->value) . '"><br>';
    // echo '<label for="voice_title">Voice Title:</label>';
    // echo '<input type="text" id="voice_title" name="voice_title" value="' . htmlspecialchars($voice->title) . '"><br>';

    // echo '<label for="age_tag">Age Tag:</label>';
    // echo '<select id="age_tag" name="age_tag">';
    // echo '<option value="10代"' . ($selected_age_tag === '10代' ? ' selected' : '') . '>10代</option>';
    // echo '<option value="20代"' . ($selected_age_tag === '20代' ? ' selected' : '') . '>20代</option>';
    // echo '<option value="30代以上"' . ($selected_age_tag === '30代以上' ? ' selected' : '') . '>30代以上</option>';
    // echo '</select><br>';

    // echo '<label for="character_tag">Character Tag:</label>';
    // echo '<select id="character_tag" name="character_tag">';
    // echo '<option value="大人しい"' . ($selected_character_tag === '大人しい' ? ' selected' : '') . '>大人しい</option>';
    // echo '<option value="快活"' . ($selected_character_tag === '快活' ? ' selected' : '') . '>快活</option>';
    // echo '<option value="セクシー・渋め"' . ($selected_character_tag === 'セクシー・渋め' ? ' selected' : '') . '>セクシー・渋め</option>';
    // echo '</select><br>';

    // echo '<input type="submit" value="Edit Tag">';
    // echo '</form>';
    // echo '
    // <form method="POST" action="delete_voice.php">
    //     <input type="hidden" id="voice_id" name="voice_id" value="' . htmlspecialchars($voice->voiceId->value) . '">
    //     <button type="submit">削除</button>
    // </form>
    // ';
    }

?>

    </div>
    <?php endif; ?>


</div>
</body>
</html>

