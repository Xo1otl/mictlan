<?php

namespace koemade\query\search;

use koemade\query\valueObjects;

class VoicesParams
{
    public string $title;
    /**
     * @var valueObjects\Tag[]
     */
    public array $tags;
    public int $page;

    /**
     * @param valueObjects\Tag[] $tags
     */
    public function __construct(string $title, array $tags, int $page)
    {
        $this->title = $title;
        $this->tags = $tags;
        $this->page = $page;
    }
}
