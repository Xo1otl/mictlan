<?php

namespace query;

class VoiceDetail
{
    public int $id;
    public string $title;
    /**
     * @var array{id: string, name: string}
     */
    public array $actor;
    public string $mime_type;
    public string $filename;
    public string $created_at;
    /** @var Tag[] */
    public array $tags;

    public function __construct(
        int $id,
        string $title,
        array $actor,
        string $mime_type,
        string $filename,
        string $created_at,
        array $tags
    ) {
        $this->id = $id;
        $this->title = $title;
        $this->actor = $actor;
        $this->mime_type = $mime_type;
        $this->filename = $filename;
        $this->created_at = $created_at;
        $this->tags = $tags;
    }
}

class Tag
{
    public int $id;
    public string $tag_name;
    public string $tag_category;

    public function __construct(int $id, string $tag_name, string $tag_category)
    {
        $this->id = $id;
        $this->tag_name = $tag_name;
        $this->tag_category = $tag_category;
    }
}
