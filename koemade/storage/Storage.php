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

        // アップロードディレクトリが存在しない場合は作成
        if (!is_dir("$this->webroot$this->uploadDir")) {
            mkdir("$this->webroot$this->uploadDir", 0755, true);
        }
    }

    /**
     * ファイルをアップロードし、正確な情報を返す
     *
     * @param array $file $_FILES['name'] の配列
     * @return array|null アップロードされたファイルの情報（path, type, size）またはnull
     */
    public function upload(array $file): ?array
    {
        // ファイルアップロードエラーの確認
        if (!isset($file['error']) || $file['error'] !== UPLOAD_ERR_OK) {
            return null;
        }

        // ファイルのMIMEタイプをサーバー側で確認
        $fileInfo = new \finfo(FILEINFO_MIME_TYPE);
        $mime_type = $fileInfo->file($file['tmp_name']);

        // ファイルサイズをサーバー側で確認
        $file_size = filesize($file['tmp_name']);

        // 安全なファイル名を生成
        $extension = pathinfo($file['name'], PATHINFO_EXTENSION);
        $uuid = uniqid('', true); // 一意のIDを生成
        $filename = "$uuid.$extension";
        $filepath = "{$this->uploadDir}$filename";

        // ファイルを移動
        if (move_uploaded_file($file['tmp_name'], "$this->webroot$filepath")) {
            return [
                'path' => $filepath,
                'type' => $mime_type, // サーバー側で確認したMIMEタイプ
                'size' => $file_size, // サーバー側で確認したファイルサイズ
            ];
        }

        return null;
    }
}
