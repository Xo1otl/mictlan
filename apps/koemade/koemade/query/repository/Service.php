<?php

namespace koemade\query\repository;

interface Service
{
    public function findVoiceWithTagsByID(string $voice_id): VoiceWithTags;
    public function actorFeed(string $actor_id): ActorFeed;
    /**
     * @return Tag[]
     */
    public function listAllTags(): array;
    /**
     * TODO: このメソッドの戻り値を定義してください
     * @return array
     */
    public function listAllAccounts(): array;
}
