<?php

namespace mysql;

class QueryRepo implements \query\Repo
{
    function actorVoices(): \query\ActorVoices
    {
        return new \query\ActorVoices();
    }
    function actors(): \query\Actors
    {
        return new \query\Actors();
    }
    function actorProfile(): \query\ActorProfile
    {
        return new \query\ActorProfile();
    }
}
