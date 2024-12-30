<?php

namespace koemade\query;

interface TaggedVoiceRepository
{
    public function findTaggedVoiceWithTagsByID($query): array;
    public function taggedVoicesOnlyActorFeed($query): array;
}
