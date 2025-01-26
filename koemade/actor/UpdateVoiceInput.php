<?php

namespace koemade\actor;

class UpdateVoiceInput
{
    public string $sub;
    public string $voice_id;
    public ?string $newTitle;
    /**
     * @var ?int[] $tagIds
     */
    public ?array $tagIds;

    public function __construct(string $sub, string $voice_id, ?string $newTitle = null, ?array $tagIds = null)
    {
        $this->sub = $sub;
        $this->voice_id = $voice_id;
        $this->newTitle = $newTitle;

        // tagIdsがnullでない場合、型チェックを行う
        if ($tagIds !== null) {
            foreach ($tagIds as $tagId) {
                if (!is_int($tagId)) {
                    throw new \InvalidArgumentException('tagIds must be an array of integers or null.');
                }
            }
        }

        $this->tagIds = $tagIds;
    }
}
