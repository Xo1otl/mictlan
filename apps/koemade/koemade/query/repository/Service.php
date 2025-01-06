<?php

namespace koemade\query\repository;

interface Service
{
    public function findVoiceWithTagsByID(string $voice_id): VoiceWithTags;
    public function actorFeed(string $actor_id): ActorFeed;
}
