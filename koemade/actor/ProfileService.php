<?php

namespace koemade\actor;

interface ProfileService
{
    public function save(string $actor_id, ProfileInput $input);
}
