<?php

namespace query;

class ActorVoice
{
    public int $id;
    public string $name;
    public string $source_url;
    /** @var string[] */
    public array $tags;
    /** @var array{overall: float, clarity: float, naturalness: float} */
    public array $ratings;
    /** @var array{id: int, name: string, status: string, rank: string, total_voices: int} */
    public array $actor;

    public function __construct(
        int $id,
        string $name,
        string $source_url,
        array $tags,
        array $ratings,
        array $actor
    ) {
        $this->id = $id;
        $this->name = $name;
        $this->source_url = $source_url;
        $this->tags = $tags;
        $this->ratings = $ratings;
        $this->actor = $actor;
    }
}
