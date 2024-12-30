<?php

namespace query;

class Price
{
    /**
     * @var int Default price
     */
    public int $default;

    /**
     * @var int Price for NSFW content
     */
    public int $nsfw;

    /**
     * @var int Price for extreme NSFW content
     */
    public int $nsfwExtreme;

    /**
     * Price constructor.
     *
     * @param int $default
     * @param int $nsfw
     * @param int $nsfwExtreme
     */
    public function __construct(int $default, int $nsfw, int $nsfwExtreme)
    {
        $this->default = $default;
        $this->nsfw = $nsfw;
        $this->nsfwExtreme = $nsfwExtreme;
    }
}

class SampleVoice
{
    /**
     * @var int Sample voice ID
     */
    public int $id;

    /**
     * @var string Sample voice name
     */
    public string $name;

    /**
     * @var string URL to the sample voice audio file
     */
    public string $sourceUrl;

    /**
     * @var array Tags associated with the sample voice
     */
    public array $tags;

    /**
     * SampleVoice constructor.
     *
     * @param int $id
     * @param string $name
     * @param string $sourceUrl
     * @param array $tags
     */
    public function __construct(
        int $id,
        string $name,
        string $sourceUrl,
        array $tags
    ) {
        $this->id = $id;
        $this->name = $name;
        $this->sourceUrl = $sourceUrl;
        $this->tags = $tags;
    }
}

class ActorProfile
{
    /**
     * @var int Actor's ID
     */
    public int $id;

    /**
     * @var string Actor's name
     */
    public string $name;

    /**
     * @var string Actor's status (e.g., "enabled", "disabled")
     */
    public string $status;

    /**
     * @var string Actor's rank (e.g., "beginner", "intermediate", "expert")
     */
    public string $rank;

    /**
     * @var string Actor's description
     */
    public string $description;

    /**
     * @var string URL to the actor's avatar image
     */
    public string $avatarUrl;
    public NSFWOptions $nsfwOptions;

    /**
     * @var Price Actor's pricing information
     */
    public Price $price;

    /**
     * @var SampleVoice[] List of sample voices
     */
    public array $sampleVoices;

    /**
     * ActorProfile constructor.
     *
     * @param int $id
     * @param string $name
     * @param string $status
     * @param string $rank
     * @param string $description
     * @param string $avatarUrl
     * @param NSFWOptions $nsfwOptions
     * @param Price $price
     * @param SampleVoice[] $sampleVoices
     */
    public function __construct(
        int $id,
        string $name,
        string $status,
        string $rank,
        string $description,
        string $avatarUrl,
        NSFWOptions $nSFWOptions,
        Price $price,
        array $sampleVoices
    ) {
        $this->id = $id;
        $this->name = $name;
        $this->status = $status;
        $this->rank = $rank;
        $this->description = $description;
        $this->avatarUrl = $avatarUrl;
        $this->nsfwOptions = $nSFWOptions;
        $this->price = $price;
        $this->sampleVoices = $sampleVoices;
    }
}
