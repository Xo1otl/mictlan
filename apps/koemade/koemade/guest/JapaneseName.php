<?php

namespace koemade\guest;

class JapaneseName
{
    public string $name;
    public string $furigana;

    public function __construct(string $name, string $furigana)
    {
        $this->name = $name;
        $this->furigana = $furigana;
    }
}