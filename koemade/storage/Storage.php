<?php

namespace koemade\storage;

class Storage
{
    private $uploadDir;
    private $webroot;

    public function __construct()
    {
        $this->webroot = '../../ui';
        $this->uploadDir = '/uploads/'; // webrootからのuploadDirのpath

        if (!is_dir($this->uploadDir)) {
            mkdir("$this->webroot$this->uploadDir", 0755, true);
        }
    }

    public function upload($file): ?array
    {
        if (isset($file['error']) && $file['error'] === UPLOAD_ERR_OK) {
            $extension = pathinfo($file['name'], PATHINFO_EXTENSION);
            $uuid = uniqid('', true); // 一意のIDを生成
            $filepath = "{$this->uploadDir}$uuid.$extension";

            if (move_uploaded_file($file['tmp_name'], "$this->webroot$filepath")) {
                return [
                    'path' => $filepath,
                    'type' => $file['type'],
                    'size' => $file['size'],
                ];
            }
        }

        return null;
    }
}
