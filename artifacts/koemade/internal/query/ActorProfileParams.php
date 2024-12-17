<?php

namespace query;

class ActorProfileParams
{
    public int $id;
    public int $page;

    public function __construct(int $id, int $page)
    {
        $this->id = $id;
        $this->page = $page;
    }
}
