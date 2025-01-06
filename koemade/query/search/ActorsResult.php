<?php

namespace koemade\query\search;

class ActorsResult
{
    public int $id;
    public string $name;
    public string $status;
    public string $rank;
    public string $avatar_url;

    public function __construct(int $id, string $name, string $status, string $rank, string $avatar_url)
    {
        $this->id = $id;
        $this->name = $name;
        $this->status = $status;
        $this->rank = $rank;
        $this->avatar_url = $avatar_url;
    }
}