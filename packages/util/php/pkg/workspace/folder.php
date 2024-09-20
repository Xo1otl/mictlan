<?php

namespace workspace;

function folder()
{
    $path = __DIR__;
    while ($path !== '/' && $path !== '') {
        if (file_exists($path . DIRECTORY_SEPARATOR . 'workspace.php')) {
            return $path;
        }
        $path = dirname($path);
    }
    throw new \Exception("Workspace root (containing workspace.php) not found.");
}
