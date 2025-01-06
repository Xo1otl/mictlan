<?php

namespace koemade\query\valueObjects;

class Price
{
    public int $default;
    public int $nsfw;
    public int $nsfw_extreme;

    public function __construct(int $default, int $nsfw, int $nsfw_extreme)
    {
        $this->default = $default;
        $this->nsfw = $nsfw;
        $this->nsfw_extreme = $nsfw_extreme;
    }
}
