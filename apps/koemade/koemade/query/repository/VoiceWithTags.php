<?php

namespace koemade\query\repository;

use koemade\query\valueObjects;

class VoiceWithTags
{
    public string $id;
    public string $title;
    /**
     * @var array{id: string, username: string, avator_url: string}
     */
    public array $account;
    public string $filename;
    public string $created_at;
    /**
     * @var valueObjects\Tag[]
     */
    public array $tags;
}
