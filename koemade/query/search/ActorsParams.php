<?php

namespace koemade\query\search;

class ActorsParams
{
    public ?string $name_like;
    public ?string $status;

    public ?bool $nsfw_allowed;
    public ?bool $nsfw_extreme_allowed;
    public ?int $page;

    /**
     * @param array{allowed?: bool, extreme_allowed?: bool} $nsfw_options 
     */
    public function __construct(?string $name_like, ?string $status, ?int $page, array $nsfw_options = [])
    {
        $this->name_like = $name_like;
        $this->nsfw_allowed = $nsfw_options['allowed'] ?? null;
        $this->nsfw_extreme_allowed = $nsfw_options['extreme_allowed'] ?? null;
        $this->status = $status;
        $this->page = $page ?? 0;
    }
}
