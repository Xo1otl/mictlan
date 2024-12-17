<?php

namespace query;

interface Repo
{
    /**
     * @param ActorVoiceParams $input
     * @return ActorVoice[]
     */
    function actorVoices(ActorVoiceParams $input): array;

    /**
     * @param ActorParams $input
     * @return Actor[]
     */
    function actors(ActorParams $input): array;

    /**
     * @param ActorProfileParams $input
     * @return ActorProfile
     */
    function actorProfile(ActorProfileParams $input): ActorProfile;
}
