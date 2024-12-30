<?php

namespace query;

class ActorVoiceParams
{
    public string $keyword;
    public string $status;
    public string $sex;
    public string $tag;
    public string $age;
    public string $delivery;
    public int $page;

    public function __construct(
        string $keyword,
        string $status,
        string $sex,
        string $tag,
        string $age,
        string $delivery,
        int $page
    ) {
        $this->keyword = $keyword;
        $this->status = $status;
        $this->sex = $sex;
        $this->tag = $tag;
        $this->age = $age;
        $this->delivery = $delivery;
        $this->page = $page;
    }
}
