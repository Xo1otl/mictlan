<?php

namespace voice;

use common\Id;

class Voice
{
    public \common\Id $voiceId;
    public \common\Id $accountId;
    public string $filename;
    public string $title;
    public string $mimeType;
    public \DateTime $createdAt;
    /**
     * @var Tag[]
     */
    public array $tags;

    private const ALLOWED_MIME_TYPES = [
        'audio/mpeg' => 'mp3',
        'audio/wav' => 'wav',
        'audio/ogg' => 'ogg',
    ];

    /**
     * @param Id $voiceId
     * @param Id $accountId
     * @param string $filename
     * @param string $title
     * @param string $mimeType
     * @param \DateTime $createdAt
     * @param Tag[] $tags
     */
    public function __construct(\common\Id $voiceId, \common\Id $accountId, string $filename, string $title, string $mimeType, \DateTime $createdAt, array $tags)
    {
        if (!array_key_exists($mimeType, self::ALLOWED_MIME_TYPES)) {
            throw new \InvalidArgumentException("Invalid mime type: $mimeType");
        }
        $this->voiceId = $voiceId;
        $this->accountId = $accountId;
        $this->filename = $filename;
        $this->title = $title;
        $this->mimeType = $mimeType;
        $this->createdAt = $createdAt;
        $this->tags = $tags;
    }

    public function getFullname(): string
    {
        $extension = self::ALLOWED_MIME_TYPES[$this->mimeType] ?? 'unknown';
        return "{$this->filename}.{$extension}";
    }
}
