<?php

namespace query;

class Handler
{
    private $repo;

    function __construct(Repo $repo)
    {
        $this->repo = $repo;
    }

    function actorVoices(): ActorVoices
    {
        return $this->repo->actorVoices();
    }

    function actors(): Actors
    {
        return $this->repo->actors();
    }

    function actorProfile(): ActorProfile
    {
        return $this->repo->actorProfile();
    }
}
