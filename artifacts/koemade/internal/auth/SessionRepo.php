<?php

namespace auth;

interface SessionRepo
{
    public function get(): ?Session;

    public function set(Session $session);
}
