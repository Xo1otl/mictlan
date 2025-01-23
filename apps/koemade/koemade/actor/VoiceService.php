<?php

namespace koemade\actor;

interface VoiceService
{
    public function editTitle(string $voice_id, string $newTitle): void;
    /**
     * @param int[] $tagIds
     */
    public function updateTags(string $voice_id, array $tagIds): void;
}
