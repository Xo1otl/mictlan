<?php

namespace filesystem;

class VoiceStorage implements \voice\Storage
{
    function uploadVoice(string $tmpPath, \voice\Input $input)
    {
        $uploadDir = '../../uploads/voices/';
        $fullPath = $uploadDir . $input->getFullname();
        $directory = dirname($fullPath);

        if (!is_dir($directory)) {
            if (!mkdir($directory, 0777, true) && !is_dir($directory)) {
                throw new \RuntimeException(sprintf('Directory "%s" was not created', $directory));
            }
        }

        if (!move_uploaded_file($tmpPath, $fullPath)) {
            throw new \RuntimeException("Failed to move voice");
        }
    }
}