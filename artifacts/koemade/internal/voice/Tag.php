<?php

namespace voice;

class Tag
{
    public string $category;
    public string $name;

    function __construct(string $category, string $name)
    {
        $this->name = $name;
        $this->category = $category;
    }
}
