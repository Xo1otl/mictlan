<?php

namespace koemade\query\search;

interface Service
{
    /**
     * @return ActorsResult[]
     */
    public function actors(ActorsParams $params): array;
    /**
     * @return VoicesResult[]
     */
    public function voices(VoicesParams $query): array;
}
