<?php

namespace query;

interface Repo
{
    function actorVoices(): ActorVoices;
    function actors(): Actors;
    function actorProfile(): ActorProfile;
}
