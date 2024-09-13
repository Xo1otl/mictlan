<?php

namespace voice;

class Input
{
    private const ALLOWED_MIME_TYPES = [
        'audio/mpeg' => 'mp3',
        'audio/wav' => 'wav',
        'audio/ogg' => 'ogg',
    ];

    public \common\Id $accountId;
    public string $title;
    public string $mimeType;
    public int $size;
    public string $filename;
    public \DateTime $createdAt;

    public function __construct(
        \common\Id $accountId,
        string     $title,
        string     $filename,
        string     $mimeType,
        int        $size,
        \DateTime  $createdAt
    )
    {
        $this->accountId = $accountId;
        $this->title = $title;
        $this->setFilename($filename);
        $this->setMimeType($mimeType);
        $this->setSize($size);
        $this->createdAt = $createdAt;
    }

    public function setFilename(string $filename)
    {
        // Validate filename to not contain path information
        if (preg_match('/[\/\\\\]/', $filename)) {
            throw new \InvalidArgumentException("Filename cannot contain path information.");
        }
        $this->filename = $filename;
    }

    public function setMimeType(string $mimeType)
    {
        if (!array_key_exists($mimeType, self::ALLOWED_MIME_TYPES)) {
            throw new \InvalidArgumentException("Invalid mime type: $mimeType");
        }
        $this->mimeType = $mimeType;
    }

    public function setSize(int $size)
    {
        // Validate size to not exceed 50MB
        if ($size > 50 * 1024 * 1024) { // 50MB in bytes
            throw new \InvalidArgumentException("File size cannot exceed 50MB.");
        }
        $this->size = $size;
    }

    public function getFullname(): string
    {
        $extension = self::ALLOWED_MIME_TYPES[$this->mimeType] ?? 'unknown';
        return "{$this->filename}.{$extension}";
    }
}
