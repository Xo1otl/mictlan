<?php

namespace koemade\query\repository;

use koemade\query\valueObjects;


class SampleVoice
{
    public string $id;
    public string $name;
    public string $source_url;
    /**
     * @var Tag[]
     */
    public array $tags;
}