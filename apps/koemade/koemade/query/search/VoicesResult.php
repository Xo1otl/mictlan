<?php

namespace koemade\query\search;

use koemade\query\valueObjects;

class VoicesResult
{
    public string $id;
    public string $name;
    public VoicesResultActor $actor;
    /**
     * @var valueObjects\Tag[]
     */
    public array $tags;
    public string $source_url;
}
