<?php

namespace koemade\actor;

interface QueryService
{
    public function userFeed(string $username): array;
}
