<?php

namespace voice;

class EditTagInput
{
    public \common\Id $accountId;
    public \common\Id $voiceId;
    public string $title;
    public string $ageTag;
    public string $characterTag;

    public function __construct(\common\Id $accountId, \common\Id $voiceId, string $title, string $ageTag, string $characterTag)
    {
        // Validate AgeTag
        // if (!in_array($ageTag, ['10代', '20代', '30代以上'])) {
        //     throw new \InvalidArgumentException('Invalid age tag. Valid tags are 10代, 20代, 30代.');
        // }

        // Validate CharacterTag
        // if (!in_array($characterTag, ['大人しい', '快活', 'セクシー・渋め'])) {
        //     throw new \InvalidArgumentException('Invalid character tag. Valid tags are 大人しい, 快活, セクシー・渋め.');
        // }

        $this->accountId = $accountId;
        $this->voiceId = $voiceId;
        $this->title = $title;
        $this->ageTag = $ageTag;
        $this->characterTag = $characterTag;
    }
}
