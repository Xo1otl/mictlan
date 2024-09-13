<?php

namespace filesystem;

class ActorStorage implements \actor\Storage
{
    function uploadProfileImageFromTmp(string $tmpPath, \actor\ProfileImage $profileImage)
    {
        $uploadDir = '../../uploads/actor/';
        $fullPath = $uploadDir . $profileImage->getFullname();
        $directory = dirname($fullPath);

        // ディレクトリが存在しない場合は再帰的に作成する
        if (!is_dir($directory)) {
            if (!mkdir($directory, 0777, true) && !is_dir($directory)) {
                throw new \RuntimeException(sprintf('Directory "%s" was not created', $directory));
            }
        }

        // ファイルを移動する
        if (!move_uploaded_file($tmpPath, $fullPath)) {
            throw new \RuntimeException("Failed to move profile image");
        }
    }
}
