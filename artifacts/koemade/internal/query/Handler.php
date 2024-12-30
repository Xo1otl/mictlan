<?php

namespace query;

class Handler
{
    private $repo;

    function __construct(Repo $repo)
    {
        $this->repo = $repo;
    }

    /**
     * @param ActorVoiceParams $input
     * @return ActorVoice[]
     */
    function actorVoices(ActorVoiceParams $input): array
    {
        return $this->repo->actorVoices($input);
    }

    /**
     * @param ActorParams $input
     * @return Actor[]
     */
    function actors(ActorParams $input): array
    {
        return $this->repo->actors($input);
    }

    function actorProfile(ActorProfileParams $input): ActorProfile
    {
        return $this->repo->actorProfile($input);
    }
}
