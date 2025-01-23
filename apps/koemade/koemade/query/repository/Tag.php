<?php

namespace koemade\query\repository;

class Tag
{
    public string $id;
    public string $category;
    public string $name;

    public function __construct(string $id, string $category, string $name)
    {
        $this->id = $id;
        $this->category = $category;
        $this->name = $name;
    }
}
