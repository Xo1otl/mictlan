<?php

namespace koemade\query\valueObjects;

class Tag
{
    public string $category;
    public string $name;

    public function __construct(string $category, string $name)
    {
        $this->category = $category;
        $this->name = $name;
    }
}
