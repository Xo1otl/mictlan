<?php

namespace koemade\query\search;

class ActorsParams
{
    public string $name_like;
    public string $status;
    public int $page;

    public function __construct(string $name_like, string $status, int $page)
    {
        $this->name_like = $name_like;
        $this->status = $status;
        $this->page = $page;
    }
}
