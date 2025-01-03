<?php

namespace koemade\actor;

interface QueryService
{
    // TODO: クエリを定義する、プロフィール画面表示用のデータ一括取得があれば十分そう
    public function userFeed(string $username): array;
}
