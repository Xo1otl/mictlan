<?php

namespace submission;

/**
 * 名前とふりがなを扱う仕様だったので日本語の名前
 *
 * 氏名に分けるかもしれない
 */
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
