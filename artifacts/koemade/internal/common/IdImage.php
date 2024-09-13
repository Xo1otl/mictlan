<?php

namespace common;

class IdImage
{
    public string $mimeType;
    public int $size;
    public string $filename;
    public \DateTime $createdAt;

    private static array $mimeToExt = [
        'image/jpeg' => 'jpg',
        'image/png' => 'png',
    ];

    private const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png'];
    private const MAX_SIZE = 3145728; // 3MB in bytes

    public function __construct(
        string    $filename,
        string    $mimeType,
        int       $size,
        \DateTime $createdAt
    )
    {
        $this->setFilename($filename);
        $this->setMimeType($mimeType);
        $this->setSize($size);
        $this->createdAt = $createdAt;
    }

    private function setFilename(string $filename)
    {
        if (!preg_match('/^[A-Za-z0-9]+$/', $filename)) {
            throw new \InvalidArgumentException("Invalid filename: $filename");
        }
        $this->filename = $filename;
    }

    public function setMimeType(string $mimeType): void
    {
        if (!in_array($mimeType, self::ALLOWED_MIME_TYPES)) {
            throw new \InvalidArgumentException("Invalid mime type. Only jpg, jpeg, and png are allowed.");
        }
        $this->mimeType = $mimeType;
    }

    public function setSize(int $size): void
    {
        if ($size > self::MAX_SIZE) {
            throw new \InvalidArgumentException("File size exceeds the maximum limit of 3MB.");
        }
        $this->size = $size;
    }

    public function getExt(): string
    {
        if (!isset(self::$mimeToExt[$this->mimeType])) {
            throw new \InvalidArgumentException("Invalid MIME type: " . $this->mimeType);
        }
        return self::$mimeToExt[$this->mimeType];
    }

    public function getFullname(): string
    {
        return $this->filename . '.' . $this->getExt();
    }
}
