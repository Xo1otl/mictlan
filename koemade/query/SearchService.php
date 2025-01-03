<?php

namespace koemade\query;

class ActorsSearchParams
{
    public string $display_name;
    public string $status;
    public int $page;
}

class ActorsSearchResult
{
    public int $id;
    public string $name;
    public string $status;
    public string $rank;
    public string $avatar_url;
}

class TaggedVoicesWithActorSearchParams
{
    public string $title;
    /**
     * @var array{category: string, name: string}[]
     */
    public array $tags;
    public int $page;
}

class TaggedVoicesWithActorSearchResult
{
    public string $title;
    /**
     * @var array{category: string, name: string}[]
     */
    public array $tags;
    public int $page;
}

interface SearchService
{
    /**
     * @return ActorsSearchResult[]
     */
    public function actors(ActorsSearchParams $params): array;
    public function taggedVoicesWithActor($query): array;
}
