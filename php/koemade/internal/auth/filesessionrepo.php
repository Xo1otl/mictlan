<?php

namespace auth;

class FileSessionRepo implements SessionRepo
{
    public function __construct()
    {
        if (session_status() === PHP_SESSION_NONE) {
            session_start();
        }
    }

    /**
     * @inheritDoc
     */
    public function get(): Session|null
    {
        if (!isset($_SESSION['session'])) {
            return null;
        }

        $session = unserialize($_SESSION['session']);

        if ($session instanceof Session) {
            return $session;
        }

        return null;
    }

    /**
     * @inheritDoc
     */
    public function set(Session $session)
    {
        $_SESSION['session'] = serialize($session);
    }

    public function delete()
    {
        $_SESSION['session'] = null;
    }
}
