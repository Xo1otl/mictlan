<?php

namespace koemade\query\repository;

use koemade\query\valueObjects;

class Actor
{
    public string $id;
    public string $name;
    public string $status;
    public string $rank;
    public string $description;
    public string $avator_url;
    public valueObjects\Price $price;
}