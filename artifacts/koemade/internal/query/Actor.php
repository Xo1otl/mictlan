<?php

namespace query;

class Actor
{
    /**
     * @var int Actor ID
     */
    public int $id;

    /**
     * @var string Actor name
     */
    public string $name;

    /**
     * @var string Acceptance status (enabled or disabled)
     */
    public string $status;

    /**
     * @var string Actor rank (e.g., expert, intermediate)
     */
    public string $rank;
    /**
     * @var string URL of the actor's avatar image
     */
    public string $avatar_url;
    public NSFWOptions $nsfwOptions;


    /**
     * Actor constructor.
     * @param int $id
     * @param string $name
     * @param string $status
     * @param string $rank
     * @param string $avatar_url
     * @param NSFWOptions $nsfwOptions 
     */
    public function __construct(int $id, string $name, string $status, string $rank, string $avatar_url, NSFWOptions $nsfwOptions)
    {
        $this->id = $id;
        $this->name = $name;
        $this->status = $status;
        $this->rank = $rank;
        $this->avatar_url = $avatar_url;
        $this->nsfwOptions = $nsfwOptions;
    }
}
