<?php

namespace koemade\query\repository;

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