<?php

namespace koemade\actor;

class NewVoiceInput
{
    public string $sub;
    public string $title;
    public string $mime_type;
    public string $path;
    /**
     * 
     * @var ?int[] $tagIds
     */
    public ?array $tagIds;

    public function __construct(string $sub, string $title, int $size, string $mime_type, string $path, ?array $tagIds = null)
    {
        $this->validateMimeType($mime_type);
        $this->validateFileSize($size); // ファイルサイズをバリデーション

        $this->sub = $sub;
        $this->title = $title;
        $this->mime_type = $mime_type;
        $this->path = $path;
        $this->tagIds = $tagIds;
    }

    private function validateMimeType(string $mime_type): void
    {
        $allowed_mime_types = [
            'audio/mpeg', // MP3
            'audio/wav',  // WAV
            'audio/ogg',  // OGG
            'audio/x-m4a', // M4A
            'audio/aac',  // AAC
        ];

        if (!in_array($mime_type, $allowed_mime_types)) {
            throw new \InvalidArgumentException("Invalid mime type. Allowed types are: " . implode(', ', $allowed_mime_types));
        }
    }

    private function validateFileSize(int $file_size): void
    {
        $max_size = 5 * 1024 * 1024; // 5MB

        if ($file_size > $max_size) {
            throw new \InvalidArgumentException("File size exceeds the maximum allowed size of 5MB.");
        }
    }
}
