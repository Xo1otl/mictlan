<?php

namespace koemade\actor;

interface VoiceService
{
    /**
     * @param int[] $tagIds
     */
    public function updateVoice(UpdateVoiceInput $updateVoiceInput): void;

    public function newVoice(NewVoiceInput $input): void;
    public function deleteVoice(string $sub, string $voice_id): void;
}
