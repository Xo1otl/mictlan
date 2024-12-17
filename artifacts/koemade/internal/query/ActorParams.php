<?php

namespace query;

class ActorParams
{
    public string $keyword;
    public string $status;
    public string $sex;
    public string $rating;
    public string $age;
    public string $delivery;
    public int $page;

    public function __construct(
        string $keyword,
        string $status,
        string $sex,
        string $rating,
        string $age,
        string $delivery,
        int $page
    ) {
        $this->keyword = $keyword;
        $this->status = $status;
        $this->sex = $sex;
        $this->rating = $rating;
        $this->age = $age;
        $this->delivery = $delivery;
        $this->page = $page;
    }
}
