<?php

namespace mysql;

class ProfileRepo implements \actor\ProfileRepo
{
    private \mysqli $mysqli;

    public function __construct()
    {
        $this->mysqli = DbConnection::getConnection();
    }

    public function addOrEdit(\actor\ProfileInput $input)
    {
        $stmt = $this->mysqli->prepare(
            "INSERT INTO profiles (display_name, category, self_promotion, price, account_id)
         VALUES (?, ?, ?, ?, ?)
         ON DUPLICATE KEY UPDATE
         display_name = VALUES(display_name),
         category = VALUES(category),
         self_promotion = VALUES(self_promotion),
         price = VALUES(price)"
        );

        if ($stmt === false) {
            \logger\fatal("Prepare failed: (" . $this->mysqli->errno . ") " . $this->mysqli->error);
        }

        $stmt->bind_param(
            'sssii',
            $input->displayName,
            $input->category,
            $input->selfPromotion,
            $input->price,
            $input->accountId->value
        );

        if (!$stmt->execute()) {
            \logger\fatal("Execute failed: (" . $stmt->errno . ") " . $stmt->error);
        }

        $stmt->close();
    }

    /**
     * @throws \Exception
     */
    public function addOrEditR(\actor\RInput $input)
    {
        $stmt = $this->mysqli->prepare("
            INSERT INTO actors_r (ok, price, hard_ok, hard_surcharge, account_id)
            VALUES (?, ?, ?, ?, ?)
            ON DUPLICATE KEY UPDATE
                ok = VALUES(ok),
                price = VALUES(price),
                hard_ok = VALUES(hard_ok),
                hard_surcharge = VALUES(hard_surcharge)
        ");

        if ($stmt === false) {
            throw new \Exception("Failed to prepare statement: " . $this->mysqli->error);
        }

        $stmt->bind_param(
            "iiiii",
            $input->ok,
            $input->price,
            $input->hardOk,
            $input->hardSurcharge,
            $input->accountId->value
        );

        if (!$stmt->execute()) {
            throw new \Exception("Failed to execute statement: " . $stmt->error);
        }

        $stmt->close();
    }

    /**
     * @throws \Exception
     */
    public function findR(\common\Id $accountId): \actor\R
    {
        $stmt = $this->mysqli->prepare("SELECT ok, price, hard_ok, hard_surcharge FROM actors_r WHERE account_id = ?");
        $stmt->bind_param("s", $accountId->value); // Assuming $accountId->getValue() returns the actual ID

        $stmt->execute();
        $stmt->bind_result($ok, $price, $hardOk, $hardSurcharge);

        if ($stmt->fetch()) {
            $result = new \actor\R($ok, $price, $hardOk, $hardSurcharge);
        } else {
            // Handle case where no result is found, e.g., by throwing an exception
            throw new \Exception("No record found for accountId: " . $accountId->value);
        }

        $stmt->close();
        return $result;
    }

    /**
     * @throws \Exception
     */
    public function findProfileImage(\common\Id $accountId): \actor\ProfileImage
    {
        $stmt = $this->mysqli->prepare("SELECT path, mime_type, size, created_at FROM profile_images WHERE account_id = ?");
        $stmt->bind_param("s", $accountId->value); // Assuming $accountId->getValue() returns the actual ID

        $stmt->execute();
        $stmt->bind_result($filename, $mimeType, $size, $createdAt);

        if ($stmt->fetch()) {
            $createdAt = \DateTime::createFromFormat('Y-m-d H:i:s', $createdAt);
            $result = new \actor\ProfileImage($accountId, $filename, $mimeType, $size, $createdAt);
        } else {
            // Handle case where no result is found, e.g., by throwing an exception
            throw new \Exception("No record found for accountId: " . $accountId->value);
        }

        $stmt->close();
        return $result;
    }

    public function findById(\common\Id $accountId): \actor\Profile
    {
        // Assuming you have a MySQLi connection in $this->mysqli
        $sql = '
        SELECT 
            p.display_name, 
            p.category, 
            p.self_promotion, 
            p.price,
            r.ok, 
            r.price AS r_price, 
            r.hard_ok, 
            r.hard_surcharge,
            pi.mime_type AS profile_image_mime_type, 
            pi.size AS profile_image_size, 
            pi.path AS profile_image_path, 
            pi.created_at AS profile_image_created_at
        FROM profiles p 
        LEFT JOIN actors_r r ON p.account_id = r.account_id 
        LEFT JOIN profile_images pi ON p.account_id = pi.account_id
        WHERE p.account_id = ?';

        $stmt = $this->mysqli->prepare($sql);

        if ($stmt === false) {
            throw new \RuntimeException('MySQLi prepare statement failed: ' . $this->mysqli->error);
        }

        $stmt->bind_param('i', $accountId->value);
        $stmt->execute();

        $stmt->bind_result(
            $displayName, $category, $selfPromotion, $price,
            $ok, $rPrice, $hardOk, $hardSurcharge, $mimeType, $size, $filename, $createdAt
        );

        if (!$stmt->fetch()) {
            throw new \RuntimeException('Profile not found');
        }

        // ないときは初期値表示しておく
        $ok = $ok ?? false;
        $rPrice = $rPrice ?? 0;
        $hardOk = $hardOk ?? false;
        $hardSurcharge = $hardSurcharge ?? 0;
        $r = new \actor\R($ok, $rPrice, $hardOk, $hardSurcharge);

        $profile = new \actor\Profile($displayName, $category, $selfPromotion, $price, $r);

        // プロフィール画像があるならつける プロフィールのテーブルもLEFT JOINする
        if ($filename !== null && $mimeType !== null && $size !== null && $createdAt !== null) {
            $createdAt = \DateTime::createFromFormat('Y-m-d H:i:s', $createdAt);
            $image = new \actor\ProfileImage($accountId, $filename, $mimeType, $size, $createdAt);
            $profile->profileImage = $image;
        }

        $stmt->close();

        return $profile;
    }

    public function updateThumbnail(\actor\ProfileImage $profileImage)
    {
        $stmt = $this->mysqli->prepare("
        INSERT INTO profile_images (account_id, mime_type, size, path, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON DUPLICATE KEY UPDATE
            mime_type = VALUES(mime_type),
            size = VALUES(size),
            path = VALUES(path),
            created_at = VALUES(created_at)
        ");

        $format = $profileImage->createdAt->format('Y-m-d H:i:s');
        $stmt->bind_param(
            'ssiss',
            $profileImage->accountId->value,
            $profileImage->mimeType,
            $profileImage->size,
            $profileImage->filename,
            $format
        );

        if (!$stmt->execute()) {
            \logger\fatal('Update failed: ' . $stmt->error);
        }

        $stmt->close();
    }
}
