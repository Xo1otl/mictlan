<?php

namespace filesystem;

class SubmissionStorage implements \submission\Storage
{
    function uploadIdImageFromTmp(string $tmpPath, \common\IdImage $idImage)
    {
        $uploadDir = '../../uploads/submission/';
        $fullPath = $uploadDir . $idImage->getFullname();
        \logger\imp("uploaded image from tmp", $fullPath);
        $directory = dirname($fullPath);

        if (!is_dir($directory)) {
            if (!mkdir($directory, 0777, true) && !is_dir($directory)) {
                throw new \RuntimeException(sprintf('Directory "%s" was not created', $directory));
            }
        }

        if (!move_uploaded_file($tmpPath, $fullPath)) {
            throw new \RuntimeException("Failed to move id image");
        }
    }

    function getIdImage()
    {

    }
}
